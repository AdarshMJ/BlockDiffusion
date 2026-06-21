"""
Extra features for diffusion model input.
"""
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import PlaceHolder


class DegreeExtraFeatures:
    """Normalized degree features computed from A_t.

    Node feature  (X): d_i(A_t) / (n_valid - 1)          shape (bs, n, 1)
    Graph features (y): mean and std of normalized degree  shape (bs, 2)

    At high t, A_t is ER-like so all nodes share similar expected degree —
    the feature collapses to a near-constant and carries no structural
    signal. This is the key distinction from LSP: degree information fed
    as a denoiser input is qualitatively different from LSP's
    degree-informed prior, which bakes degree structure into A_T itself
    and makes it available from the very first reverse step.
    """

    def __call__(self, noisy_data):
        E_t = noisy_data['E_t']           # (bs, n, n, 2)
        node_mask = noisy_data['node_mask']  # (bs, n)
        bs, n = node_mask.shape

        mask2d = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)    # (bs, n, n)
        A_t = E_t[..., 1].float() * mask2d.float()                    # (bs, n, n)

        d = A_t.sum(-1)                                                # (bs, n)
        n_valid = node_mask.float().sum(-1, keepdim=True).clamp(min=2.0)  # (bs, 1)
        d_norm = (d / (n_valid - 1.0)) * node_mask.float()            # (bs, n) in [0,1]

        mean_d = (d_norm * node_mask.float()).sum(-1, keepdim=True) / n_valid  # (bs, 1)
        var_d  = ((d_norm - mean_d) ** 2 * node_mask.float()).sum(-1) / n_valid.squeeze(-1)
        std_d  = var_d.clamp(min=0.0).sqrt()                          # (bs,)
        graph_feats = torch.stack([mean_d.squeeze(-1), std_d], dim=-1)  # (bs, 2)

        empty_e = E_t.new_zeros(bs, n, n, 0)
        return PlaceHolder(
            X=d_norm.unsqueeze(-1),   # (bs, n, 1)
            E=empty_e,
            y=graph_feats,            # (bs, 2)
        )


class DummyExtraFeatures:
    """Returns empty tensors (no extra features)."""
    
    def __init__(self):
        pass

    def __call__(self, noisy_data):
        X = noisy_data['X_t']
        E = noisy_data['E_t']
        y = noisy_data['y_t']
        empty_x = X.new_zeros((*X.shape[:-1], 0))
        empty_e = E.new_zeros((*E.shape[:-1], 0))
        empty_y = y.new_zeros((y.shape[0], 0))
        return PlaceHolder(X=empty_x, E=empty_e, y=empty_y)


class SelfConditionedDiGressFeatures:
    """DiGress's structural features, computed on A_0_hat instead of A_t.

    Design principle: for a diffusion model with an informative structural
    prior (here the LogNormal sociability prior on u), features evaluated
    on the noisy A_t either duplicate what u_t already encodes or are
    dominated by forward-process noise at high t. Evaluating the same
    feature set on the denoiser's own point estimate A_0_hat avoids both
    failure modes — u_t is absorbed into A_0_hat through the network, so
    features of A_0_hat carry signal orthogonal to the prior's sufficient
    statistic by construction, at every noise level.

    Feature set: identical shapes and semantics to DiGress's ExtraFeatures
    (cycle counts, Laplacian eigenvalues/eigenvectors, normalised size).
    Only the source graph changes — pred_A0 replaces E_t.

    Training regime (caller responsibility):
        Each step, with probability p_self (recommend 0.5):
            1. Run a detached forward pass with zero-valued extras to get
               logits for A_0; softmax → pred_A0 soft one-hot.
            2. Put pred_A0 (detached) into noisy_data['pred_A0'].
            3. Run the main forward pass; this class computes features on
               pred_A0.
        Otherwise, leave noisy_data['pred_A0'] = None. This class then
        returns zero-valued features of the correct shape, forcing the
        network to still function without self-conditioning and preventing
        it from collapsing to feature shortcuts.

    Inference regime (caller responsibility):
        Maintain a running pred_A0 across reverse steps. Initialise at t=T
        from the terminal NR sample (one-hot of sampled A_T). After each
        reverse step, update pred_A0 ← softmax(pred.E) for the next step.

    Expects noisy_data to optionally contain:
        pred_A0: (bs, n, n, 2) soft one-hot estimate of A_0. When None or
                 absent, returns zero tensors of the correct shape.
    """

    def __init__(self, extra_features_type, dataset_info):
        self._underlying = ExtraFeatures(extra_features_type, dataset_info)
        self._shape_cache = None

    def _cache_shapes(self, placeholder):
        self._shape_cache = (
            placeholder.X.shape[-1],
            placeholder.E.shape[-1],
            placeholder.y.shape[-1],
        )

    def __call__(self, noisy_data):
        pred_A0 = noisy_data.get('pred_A0', None)
        E_t = noisy_data['E_t']
        bs, n = noisy_data['node_mask'].shape

        if pred_A0 is None:
            # Self-conditioning signal not available (dropout-off branch or
            # dimension probe). Return zeros of the correct shape; the
            # network learns that all-zero extras mean "no self-cond".
            if self._shape_cache is None:
                # Use the current noisy_data (with E_t) as a one-time probe
                # to discover the output shape. The values are discarded.
                with torch.no_grad():
                    probe = self._underlying(noisy_data)
                self._cache_shapes(probe)
            x_dim, e_dim, y_dim = self._shape_cache
            return PlaceHolder(
                X=E_t.new_zeros(bs, n, x_dim),
                E=E_t.new_zeros(bs, n, n, e_dim),
                y=E_t.new_zeros(bs, y_dim),
            )

        # Clean pred_A0 before feature extraction:
        #   - zero padded rows/cols (node_mask off-diagonal)
        #   - zero self-loops (diagonal)
        # so cycle counts and Laplacian spectrum are not polluted.
        node_mask = noisy_data['node_mask']
        mask2d = (node_mask.unsqueeze(2) * node_mask.unsqueeze(1)).float()
        offdiag = 1.0 - torch.eye(n, device=pred_A0.device, dtype=pred_A0.dtype)
        valid = mask2d * offdiag.unsqueeze(0)                          # (bs, n, n)
        edge_prob = pred_A0[..., 1] * valid                            # (bs, n, n)
        pred_A0_clean = torch.stack(
            [1.0 - edge_prob, edge_prob], dim=-1
        )                                                              # (bs, n, n, 2)

        noisy_data_sc = {**noisy_data, 'E_t': pred_A0_clean}
        out = self._underlying(noisy_data_sc)

        if self._shape_cache is None:
            self._cache_shapes(out)
        return out


class CRMResidualAtFeatures:
    """Residual spectral features computed from A_t with MMSE-corrected Pi reference.

    Two root causes of the naive A_t - Pi(u_t) approach failing:
      1. Degree contamination: Pi(u_t) != Pi(u_0) at intermediate t, so the
         residual picks up the degree mismatch (Pi(u_0) - Pi(u_t)) as noise.
      2. Sign instability: argmax-based sign canonicalization can flip between
         consecutive reverse steps, giving the model inconsistent community coords.

    Fixes applied here:
      - Training:   use Pi(w_0) — clean sociability from noisy_data['u_0'].
                    R = A_t - Pi(w_0) is the true community residual with zero
                    degree contamination. At high t, A_t ≈ Pi(w_0), so R ≈ 0.
      - Inference:  use Pi(w_0_est) where w_0_est comes from the closed-form MMSE
                    estimate  u_0_est = mu_pi + sqrt(alpha_bar_t) * (u_t - mu_pi).
                    No bootstrapping — derived purely from the diffusion schedule.
      - Sign:       flip each eigenvector so its sociability-weighted projection
                    sum_i w_ref_i * v_{k,i} > 0.  Stable because u_t changes
                    smoothly across reverse steps, so the dominant-community side
                    stays consistent.

    Node features (X): K scaled eigenvectors  (bs, n, K)
    Graph features (y): K eigenvalue magnitudes (bs, K)
    """

    def __init__(self, K_eig=2, c=4.0, u_clamp=15.0, t_cutoff=0.2):
        self.K_eig    = K_eig
        self.c        = c
        self.u_clamp  = u_clamp
        self.mu_pi    = 0.0   # set via set_prior_mu() after calibration
        # Features are zeroed for samples where alpha_t_bar < t_cutoff (high noise).
        # At alpha_t_bar < 0.2, MMSE estimate of u_0 is poor, R is dominated by
        # forward-process noise, and eigenvectors have random sign orientation.
        # Gating ensures train/inference consistency and prevents the model from
        # learning spurious associations with noisy residual features.
        self.t_cutoff = t_cutoff

    def set_prior_mu(self, mu):
        self.mu_pi = float(mu)

    def __call__(self, noisy_data):
        E_t       = noisy_data['E_t']        # (bs, n, n, 2)
        node_mask = noisy_data['node_mask']  # (bs, n)
        bs, n     = node_mask.shape
        device    = E_t.device

        empty_e = E_t.new_zeros(bs, n, n, 0)

        # Probe call from compute_crm_input_output_dims has no u_t — return zeros.
        if 'u_t' not in noisy_data:
            return PlaceHolder(
                X=E_t.new_zeros(bs, n, self.K_eig),
                E=empty_e,
                y=E_t.new_zeros(bs, self.K_eig),
            )

        u_t    = noisy_data['u_t']   # (bs, n)
        mask2d = (node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)).float()
        eye    = torch.eye(n, device=device).unsqueeze(0)

        # 1. Reference weights for Pi — clean u_0 at training, MMSE estimate at inference.
        #    Training noisy_data carries u_0 (not w_0); inference does not.
        if 'u_0' in noisy_data:
            w_ref = torch.exp(noisy_data['u_0'].clamp(-self.u_clamp, self.u_clamp))  # training
        else:
            # Closed-form MMSE estimate: E[u_0 | u_t] = mu_pi + sqrt(ab_t)*(u_t - mu_pi)
            alpha_t_bar = noisy_data.get('alpha_t_bar', None)
            if alpha_t_bar is not None:
                sqrt_ab = torch.sqrt(alpha_t_bar.view(bs, 1).clamp(min=0.0))
                u_0_est = self.mu_pi + sqrt_ab * (u_t - self.mu_pi)
            else:
                u_0_est = u_t                                        # fallback, probe
            w_ref = torch.exp(u_0_est.clamp(-self.u_clamp, self.u_clamp))
        w_ref = w_ref * node_mask.float()

        # 2. Rank-1 LSP prediction Pi(w_ref)
        Pi  = 1.0 - torch.exp(-self.c * w_ref.unsqueeze(-1) * w_ref.unsqueeze(-2))
        Pi  = Pi * mask2d * (1.0 - eye)                             # (bs, n, n)

        # 3. Residual: structure beyond what degree predicts
        A_t = E_t[..., 1].float() * mask2d                         # (bs, n, n)
        R   = A_t - Pi                                              # (bs, n, n)

        # 4. Degree-normalise with Pi's expected degrees
        d_Pi       = Pi.sum(-1).clamp(min=1e-6)                    # (bs, n)
        d_inv_sqrt = d_Pi.pow(-0.5) * node_mask.float()            # (bs, n)
        R_tilde    = d_inv_sqrt.unsqueeze(-1) * R * d_inv_sqrt.unsqueeze(-2)
        R_tilde    = (R_tilde + R_tilde.transpose(-1, -2)) * 0.5  # enforce symmetry
        R_tilde    = R_tilde * mask2d                               # zero padding

        # 5. Eigendecompose
        eigvals, eigvecs = torch.linalg.eigh(R_tilde)              # (bs,n), (bs,n,n)

        # 6. Top-K by |eigenvalue|
        K = min(self.K_eig, n)
        _, topk_idx = eigvals.abs().topk(K, dim=-1)                # (bs, K)
        topk_eigs   = eigvals.gather(-1, topk_idx)                 # (bs, K)
        topk_vecs   = eigvecs.gather(                              # (bs, n, K)
            -1, topk_idx.unsqueeze(1).expand(-1, n, -1))

        # 7. Sociability-weighted sign canonicalization (stable across reverse steps).
        #    Flip each eigenvector so sum_i w_ref_i * v_{k,i} > 0.
        soc_proj = (w_ref.unsqueeze(-1) * topk_vecs).sum(dim=1)   # (bs, K)
        sgn      = soc_proj.sign()
        sgn      = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        topk_vecs = topk_vecs * sgn.unsqueeze(1)                   # (bs, n, K)

        # 8. Scale by eigenvalue — features self-zero when R is near-zero (high t)
        node_feats  = topk_vecs * topk_eigs.unsqueeze(1)           # (bs, n, K)
        node_feats  = node_feats * node_mask.unsqueeze(-1)

        graph_feats = topk_eigs.abs()                               # (bs, K)

        # 9. Noise-level gate: zero features for samples above the noise cutoff.
        #    At low alpha_t_bar (high noise), R is dominated by forward-process noise,
        #    the MMSE reference Pi diverges from clean Pi, and eigenvector signs are
        #    random. The gate ensures train/inference consistency — same criterion
        #    both times — and prevents empty-graph collapse from noisy high-t features.
        ab = noisy_data.get('alpha_t_bar', None)
        if ab is not None:
            gate = (ab.view(bs, 1) >= self.t_cutoff).float()       # (bs, 1)
            node_feats  = node_feats  * gate.unsqueeze(-1)          # (bs, n, K)
            graph_feats = graph_feats * gate                         # (bs, K)

        return PlaceHolder(X=node_feats, E=empty_e, y=graph_feats)


class ExtraFeatures:
    """Compute extra features for the diffusion model.

    Supported feature types:
      'cycles'      — raw cycle counts (k3–k6), divided by 10 and clamped to 1.
      'norm_cycles' — normalized cycles: k3 replaced by local clustering coefficient
                      (bounded [0,1]); k4/k5/k6 still divided by 10 and clamped.
      'eigenvalues' — raw cycles + 5 smallest combinatorial Laplacian eigenvalues.
      'all'         — raw cycles + eigenvalues + eigenvectors (LCC indicator + k=2 vecs).
      'norm_all'    — norm_cycles + eigenvalues + eigenvectors (same spectral as 'all').
    """

    def __init__(self, extra_features_type, dataset_info):
        self.max_n_nodes = dataset_info.max_n_nodes
        normalize = extra_features_type in ['norm_cycles', 'norm_all']
        self.ncycles = NodeCycleFeatures(normalize=normalize)
        self.features_type = extra_features_type
        # Map norm_all -> 'all' for EigenFeatures mode
        eigen_mode = 'all' if extra_features_type == 'norm_all' else extra_features_type
        if extra_features_type in ['eigenvalues', 'all', 'norm_all']:
            self.eigenfeatures = EigenFeatures(mode=eigen_mode)

    def __call__(self, noisy_data):
        n = noisy_data['node_mask'].sum(dim=1).unsqueeze(1) / self.max_n_nodes
        x_cycles, y_cycles = self.ncycles(noisy_data)

        if self.features_type in ('cycles', 'norm_cycles'):
            E = noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            return PlaceHolder(X=x_cycles, E=extra_edge_attr, y=torch.hstack((n, y_cycles)))

        elif self.features_type == 'eigenvalues':
            eigenfeatures = self.eigenfeatures(noisy_data)
            E = noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            n_components, batched_eigenvalues = eigenfeatures
            return PlaceHolder(X=x_cycles, E=extra_edge_attr, y=torch.hstack((n, y_cycles, n_components,
                                                                                batched_eigenvalues)))
        elif self.features_type in ('all', 'norm_all'):
            eigenfeatures = self.eigenfeatures(noisy_data)
            E = noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            n_components, batched_eigenvalues, nonlcc_indicator, k_lowest_eigvec = eigenfeatures

            return PlaceHolder(X=torch.cat((x_cycles, nonlcc_indicator, k_lowest_eigvec), dim=-1),
                               E=extra_edge_attr,
                               y=torch.hstack((n, y_cycles, n_components, batched_eigenvalues)))
        else:
            raise ValueError(f"Features type {self.features_type} not implemented")


class NodeCycleFeatures:
    """Compute cycle counts for nodes.

    When normalize=True (used by 'norm_cycles' / 'norm_all'):
      - k3x (per-node) is replaced by the local clustering coefficient
            c_i = 2 * triangles_i / (d_i * (d_i - 1))  in [0, 1]
      - k3y (graph-level) is replaced by the mean local clustering coefficient
      - k4x, k5x, k4y, k5y, k6y are still divided by 10 and clamped to [0, 1]
    """

    def __init__(self, normalize=False):
        self.kcycles = KNodeCycles()
        self.normalize = normalize

    def __call__(self, noisy_data):
        adj_matrix = noisy_data['E_t'][..., 1:].sum(dim=-1).float()
        node_mask = noisy_data['node_mask']

        x_cycles, y_cycles = self.kcycles.k_cycles(adj_matrix=adj_matrix)
        x_cycles = x_cycles.type_as(adj_matrix) * node_mask.unsqueeze(-1)

        if self.normalize:
            d = adj_matrix.sum(dim=-1)  # (bs, n)  — zero for padded nodes

            # Local clustering coefficient: c_i = 2*k3x_i / (d_i*(d_i-1)), bounded in [0,1]
            denom_3 = (d * (d - 1)).clamp(min=1.0).unsqueeze(-1)  # (bs, n, 1)
            k3x_norm = 2.0 * x_cycles[..., 0:1] / denom_3         # (bs, n, 1)

            k4x = (x_cycles[..., 1:2] / 10).clamp(max=1.0)
            k5x = (x_cycles[..., 2:3] / 10).clamp(max=1.0)
            x_cycles = torch.cat([k3x_norm, k4x, k5x], dim=-1)

            # Mean local clustering coefficient over valid nodes (graph-level)
            n_valid = node_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            avg_clust = (k3x_norm.squeeze(-1) * node_mask.float()).sum(dim=1, keepdim=True) / n_valid
            k4y = (y_cycles[..., 1:2] / 10).clamp(max=1.0)
            k5y = (y_cycles[..., 2:3] / 10).clamp(max=1.0)
            k6y = (y_cycles[..., 3:4] / 10).clamp(max=1.0)
            y_cycles = torch.cat([avg_clust, k4y, k5y, k6y], dim=-1)
        else:
            # Avoid large values when the graph is dense
            x_cycles = x_cycles / 10
            y_cycles = y_cycles / 10
            x_cycles[x_cycles > 1] = 1
            y_cycles[y_cycles > 1] = 1

        return x_cycles, y_cycles


class EigenFeatures:
    """Compute eigenvalue-based features."""
    
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, noisy_data):
        E_t = noisy_data['E_t']
        mask = noisy_data['node_mask']
        A = E_t[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
        L = compute_laplacian(A, normalize=False)
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
        mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
        L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

        if self.mode == 'eigenvalues':
            eigvals = torch.linalg.eigvalsh(L)
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)

            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)
            return n_connected_comp.type_as(A), batch_eigenvalues.type_as(A)

        elif self.mode == 'all':
            eigvals, eigvectors = torch.linalg.eigh(L)
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
            eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)
            
            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)
            nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(
                vectors=eigvectors, node_mask=noisy_data['node_mask'], n_connected=n_connected_comp)
            return n_connected_comp, batch_eigenvalues, nonlcc_indicator, k_lowest_eigenvector
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")


def compute_laplacian(adjacency, normalize: bool):
    """Compute graph Laplacian."""
    diag = torch.sum(adjacency, dim=-1)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)
    combinatorial = D - adjacency

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)
    D_norm = torch.diag_embed(diag_norm)
    L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2


def get_eigenvalues_features(eigenvalues, k=5):
    """Get eigenvalue-based features."""
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    
    # Ensure at least one connected component (handle edge cases)
    n_connected_components = torch.clamp(n_connected_components, min=1)

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack((eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues)))
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(0) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """Get eigenvector-based features."""
    bs, n = vectors.size(0), vectors.size(1)

    first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask
    random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)
    first_ev = first_ev + random
    most_common = torch.mode(first_ev, dim=1).values
    mask = ~ (first_ev == most_common.unsqueeze(1))
    not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat((vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2)
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(0) + n_connected.unsqueeze(2)
    indices = indices.expand(-1, n, -1)
    first_k_ev = torch.gather(vectors, dim=2, index=indices)
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev


def batch_trace(X):
    """Compute trace of batched matrices."""
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X):
    """Extract diagonal from batched matrices."""
    return torch.diagonal(X, dim1=-2, dim2=-1)


class KNodeCycles:
    """Builds cycle counts for each node in a graph."""

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()

    def k3_cycle(self):
        """tr(A ** 3)."""
        c3 = batch_diagonal(self.k3_matrix)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = diag_a4 - self.d * (self.d - 1) - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(-1).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix)
        c5 = diag_a5 - 2 * triangles * self.d - (self.adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1) + triangles
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(-1).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix ** 2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (term_1_t - 3 * term_2_t + 9 * term3_t - 6 * term_4_t + 6 * term_5_t - 4 * term_6_t + 4 * term_7_t +
                3 * term8_t - 12 * term9_t + 4 * term10_t)
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix, verbose=False):
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        k4x, k4y = self.k4_cycle()
        k5x, k5y = self.k5_cycle()
        _, k6y = self.k6_cycle()
        
        # Clamp to zero to handle numerical precision issues
        # (cycle counts should theoretically be non-negative)
        k3x = torch.clamp(k3x, min=0.0)
        k3y = torch.clamp(k3y, min=0.0)
        k4x = torch.clamp(k4x, min=0.0)
        k4y = torch.clamp(k4y, min=0.0)
        k5x = torch.clamp(k5x, min=0.0)
        k5y = torch.clamp(k5y, min=0.0)
        k6y = torch.clamp(k6y, min=0.0)

        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)
        return kcyclesx, kcyclesy
