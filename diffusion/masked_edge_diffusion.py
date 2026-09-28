"""Mask-aware binary edge diffusion for factorized decoder regions.

Only upper-triangular variables selected by ``update_mask`` are noised and
scored. All other entries are immutable conditioning context. This is the key
difference from applying ordinary DiGress independently to each induced pair
subgraph: the already generated intra edges in an inter region remain clean.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete


class MaskedEdgeDiffusion(nn.Module):
    """Forward noising and masked x0 loss for binary local adjacencies."""

    def __init__(
        self,
        timesteps: int,
        edge_marginals_by_kind: torch.Tensor,
        noise_schedule: str = "cosine",
    ):
        super().__init__()
        marginals = torch.as_tensor(edge_marginals_by_kind, dtype=torch.float)
        if marginals.shape != (2, 2):
            raise ValueError("edge_marginals_by_kind must have shape (2 kinds, 2 classes)")
        if (marginals < 0).any() or not torch.allclose(
            marginals.sum(dim=-1), torch.ones(2, device=marginals.device), atol=1e-5
        ):
            raise ValueError("each edge marginal must be a probability distribution")
        if timesteps < 1:
            raise ValueError("timesteps must be positive")
        self.T = int(timesteps)
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(noise_schedule, timesteps)
        self.register_buffer("edge_marginals_by_kind", marginals)

    @staticmethod
    def _active_edge_mask(node_mask: torch.Tensor) -> torch.Tensor:
        n = node_mask.shape[1]
        off_diagonal = ~torch.eye(n, dtype=torch.bool, device=node_mask.device)
        return node_mask[:, :, None] & node_mask[:, None, :] & off_diagonal[None]

    def _validate_batch(self, batch: Mapping[str, torch.Tensor]) -> None:
        clean = batch["clean_adjacency"]
        update = batch["update_mask"]
        node_mask = batch["node_mask"]
        kind = batch["kind"]
        if clean.ndim != 3 or clean.shape[1] != clean.shape[2]:
            raise ValueError("clean_adjacency must have shape (batch, n, n)")
        if update.shape != clean.shape:
            raise ValueError("update_mask shape must match clean_adjacency")
        if node_mask.shape != clean.shape[:2]:
            raise ValueError("node_mask shape must match adjacency node axes")
        if kind.shape != (clean.shape[0],) or ((kind < 0) | (kind > 1)).any():
            raise ValueError("kind must contain one INTRA/INTER id per sample")
        if ((clean != 0) & (clean != 1)).any():
            raise ValueError("clean_adjacency must be binary")
        if not torch.equal(clean, clean.transpose(1, 2)):
            raise ValueError("clean_adjacency must be symmetric")
        if not torch.equal(update, update.transpose(1, 2)):
            raise ValueError("update_mask must be symmetric")
        if (update & ~self._active_edge_mask(node_mask)).any():
            raise ValueError("update_mask selects a diagonal or padded edge")

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample t uniformly from 1..T, one shared scalar per region."""
        return torch.randint(1, self.T + 1, (batch_size,), device=device)

    def q_sample(
        self,
        batch: Mapping[str, torch.Tensor],
        t_int: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample ``q(A_t | A_0)`` on update variables and freeze context."""
        self._validate_batch(batch)
        clean = batch["clean_adjacency"].long()
        update = batch["update_mask"].bool()
        kind = batch["kind"].long()
        batch_size, n, _ = clean.shape
        if t_int is None:
            t_int = self.sample_timesteps(batch_size, clean.device)
        t_int = t_int.to(device=clean.device, dtype=torch.long).reshape(-1)
        if t_int.shape != (batch_size,) or (t_int < 0).any() or (t_int > self.T).any():
            raise ValueError(f"t_int must have shape ({batch_size},) with values in [0, T]")

        alpha_bar = self.noise_schedule.get_alpha_bar(t_int=t_int).reshape(batch_size, 1, 1, 1)
        clean_one_hot = F.one_hot(clean, num_classes=2).float()
        marginal = self.edge_marginals_by_kind[kind].reshape(batch_size, 1, 1, 2)
        probabilities = alpha_bar * clean_one_hot + (1.0 - alpha_bar) * marginal

        noisy = clean.clone()
        upper_update = torch.triu(update, diagonal=1)
        if upper_update.any():
            sampled = torch.multinomial(probabilities[upper_update], num_samples=1).squeeze(-1)
            indices = upper_update.nonzero(as_tuple=True)
            noisy[indices] = sampled
            noisy[indices[0], indices[2], indices[1]] = sampled

        # Diagonal and padding are all-zero vectors, matching GraphTransformer's
        # existing DiGress convention rather than representing them as no-edge.
        active = self._active_edge_mask(batch["node_mask"])
        noisy_one_hot = F.one_hot(noisy, num_classes=2).float() * active.unsqueeze(-1)
        if not torch.equal(noisy, noisy.transpose(1, 2)):
            raise AssertionError("q_sample produced an asymmetric adjacency")
        if not torch.equal(noisy[~update], clean[~update]):
            raise AssertionError("q_sample changed frozen conditioning context")

        return {
            "adjacency_t": noisy,
            "E_t": noisy_one_hot,
            "t_int": t_int,
            "t": t_int.float() / self.T,
            "alpha_bar_t": alpha_bar.reshape(batch_size),
        }

    def masked_x0_loss(
        self,
        edge_logits: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Cross-entropy over each undirected update variable exactly once.

        Losses are first averaged per region and then across regions, preventing
        larger cluster pairs from dominating merely because they contain more
        candidate edges.
        """
        self._validate_batch(batch)
        clean = batch["clean_adjacency"].long()
        if edge_logits.shape != (*clean.shape, 2):
            raise ValueError("edge_logits must have shape (batch, n, n, 2)")
        upper_update = torch.triu(batch["update_mask"].bool(), diagonal=1)
        counts = upper_update.flatten(1).sum(dim=1)
        valid = counts > 0
        if not valid.any():
            raise ValueError("batch contains no denoising edge variables")

        element_loss = F.cross_entropy(
            edge_logits.reshape(-1, 2), clean.reshape(-1), reduction="none"
        ).reshape_as(clean)
        per_region = (element_loss * upper_update).flatten(1).sum(dim=1) / counts.clamp_min(1)
        loss = per_region[valid].mean()

        with torch.no_grad():
            prediction = edge_logits.argmax(dim=-1)
            correct = ((prediction == clean) & upper_update).flatten(1).sum(dim=1)
            accuracy = (correct[valid] / counts[valid]).mean()
            target_density = (
                (clean.bool() & upper_update).flatten(1).sum(dim=1)[valid] / counts[valid]
            ).mean()
        return loss, {
            "edge_accuracy": accuracy,
            "target_density": target_density,
            "edge_variables": counts[valid].sum(),
        }

    @torch.no_grad()
    def sample_limit(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Initialize update variables from the kind-specific limiting prior."""
        self._validate_batch(batch)
        adjacency = batch["clean_adjacency"].clone().long()
        upper_update = torch.triu(batch["update_mask"].bool(), diagonal=1)
        if upper_update.any():
            kind_per_edge = batch["kind"][:, None, None].expand_as(adjacency)[upper_update]
            probabilities = self.edge_marginals_by_kind[kind_per_edge]
            sampled = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
            indices = upper_update.nonzero(as_tuple=True)
            adjacency[indices] = sampled
            adjacency[indices[0], indices[2], indices[1]] = sampled
        return adjacency

    def adjacency_one_hot(
        self,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a sampled adjacency using the GraphTransformer convention."""
        active = self._active_edge_mask(node_mask)
        return F.one_hot(adjacency.long(), num_classes=2).float() * active.unsqueeze(-1)

    @torch.no_grad()
    def p_sample_step(
        self,
        *,
        adjacency_t: torch.Tensor,
        x0_logits: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        t_int: torch.Tensor,
    ) -> torch.Tensor:
        """Sample one analytic ``p_theta(A_{t-1} | A_t)`` masked step.

        The predicted x0 distribution is mixed with the exact discrete forward
        posterior. At the final t=1 step, sample x0 directly so the returned
        state is clean rather than the schedule's nearly-clean t=0 state.
        """
        self._validate_batch(batch)
        clean_context = batch["clean_adjacency"].long()
        update = batch["update_mask"].bool()
        kind = batch["kind"].long()
        batch_size = adjacency_t.shape[0]
        t_int = t_int.to(adjacency_t.device, dtype=torch.long).reshape(-1)
        if t_int.shape != (batch_size,) or (t_int < 1).any() or (t_int > self.T).any():
            raise ValueError(f"reverse t_int must have shape ({batch_size},) in [1, T]")
        if adjacency_t.shape != clean_context.shape or x0_logits.shape != (*adjacency_t.shape, 2):
            raise ValueError("reverse state/logit shape mismatch")
        if not torch.equal(adjacency_t, adjacency_t.transpose(1, 2)):
            raise ValueError("reverse adjacency_t must be symmetric")

        upper_update = torch.triu(update, diagonal=1)
        if not upper_update.any():
            return clean_context.clone()
        edge_indices = upper_update.nonzero(as_tuple=True)
        edge_batch = edge_indices[0]
        edge_t = t_int[edge_batch]
        edge_kind = kind[edge_batch]
        current_class = adjacency_t[edge_indices].long()
        p0 = F.softmax(x0_logits[edge_indices], dim=-1)
        marginals = self.edge_marginals_by_kind[edge_kind]
        identity = torch.eye(2, device=adjacency_t.device).expand(p0.shape[0], -1, -1)

        beta_t = self.noise_schedule(t_int=edge_t).reshape(-1, 1, 1)
        alpha_bar_s = self.noise_schedule.get_alpha_bar(t_int=edge_t - 1).reshape(-1, 1, 1)
        alpha_bar_t = self.noise_schedule.get_alpha_bar(t_int=edge_t).reshape(-1, 1, 1)
        marginal_matrix = marginals[:, None, :].expand(-1, 2, -1)
        Q_t = (1.0 - beta_t) * identity + beta_t * marginal_matrix
        Qbar_s = alpha_bar_s * identity + (1.0 - alpha_bar_s) * marginal_matrix
        Qbar_t = alpha_bar_t * identity + (1.0 - alpha_bar_t) * marginal_matrix

        # q(z_s=k | z_t=j, x0=i), with i and k as the two matrix axes.
        j = current_class[:, None, None]
        q_t_to_j = Q_t.gather(2, j.expand(-1, 2, 1)).squeeze(-1)  # (edges, k)
        qbar_t_to_j = Qbar_t.gather(2, j.expand(-1, 2, 1)).squeeze(-1)  # (edges, i)
        posterior = Qbar_s * q_t_to_j[:, None, :]
        posterior = posterior / qbar_t_to_j.clamp_min(1e-12)[:, :, None]
        probabilities = (p0[:, :, None] * posterior).sum(dim=1)

        final_step = edge_t == 1
        probabilities[final_step] = p0[final_step]
        probabilities = probabilities.clamp_min(0)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled = torch.multinomial(probabilities, num_samples=1).squeeze(-1)

        adjacency_s = clean_context.clone()
        adjacency_s[edge_indices] = sampled
        adjacency_s[edge_indices[0], edge_indices[2], edge_indices[1]] = sampled
        if not torch.equal(adjacency_s[~update], clean_context[~update]):
            raise AssertionError("reverse step changed frozen context")
        if not torch.equal(adjacency_s, adjacency_s.transpose(1, 2)):
            raise AssertionError("reverse step produced an asymmetric adjacency")
        return adjacency_s

    @torch.no_grad()
    def sample_regions(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
        *,
        enforce_inter_budget: bool = True,
    ) -> torch.Tensor:
        """Run the complete masked reverse chain for a batch of local regions."""
        adjacency = self.sample_limit(batch)
        batch_size = adjacency.shape[0]
        for timestep in range(self.T, 0, -1):
            t_int = torch.full(
                (batch_size,), timestep, dtype=torch.long, device=adjacency.device
            )
            noisy = {
                "adjacency_t": adjacency,
                "E_t": self.adjacency_one_hot(adjacency, batch["node_mask"]),
                "t_int": t_int,
                "t": t_int.float() / self.T,
            }
            x0_logits = model(batch, noisy)
            adjacency = self.p_sample_step(
                adjacency_t=adjacency,
                x0_logits=x0_logits,
                batch=batch,
                t_int=t_int,
            )
        if enforce_inter_budget:
            adjacency = self.project_inter_budgets(adjacency, x0_logits, batch)
        return adjacency

    @torch.no_grad()
    def project_inter_budgets(
        self,
        adjacency: torch.Tensor,
        x0_logits: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Select exactly ``w_ij`` cross-edges for each inter region.

        Gumbel-perturbed edge log-odds retain stochasticity while the top-k
        projection enforces the coarse blueprint's exact cardinality. Intra
        regions are untouched because ``e_i`` is not a generated condition.
        """
        projected = adjacency.clone()
        upper_update = torch.triu(batch["update_mask"].bool(), diagonal=1)
        for item in range(projected.shape[0]):
            if int(batch["kind"][item]) != 1:  # INTRA
                continue
            candidates = upper_update[item].nonzero(as_tuple=False)
            budget = int(batch["edge_budget"][item])
            if not 0 <= budget <= candidates.shape[0]:
                raise ValueError(
                    f"inter budget {budget} cannot fit in {candidates.shape[0]} candidate edges"
                )
            u, v = candidates[:, 0], candidates[:, 1]
            projected[item, u, v] = 0
            projected[item, v, u] = 0
            if budget:
                logits = x0_logits[item, u, v]
                log_odds = logits[:, 1] - logits[:, 0]
                uniform = torch.rand_like(log_odds).clamp_(1e-8, 1.0 - 1e-8)
                gumbel = -torch.log(-torch.log(uniform))
                selected = torch.topk(log_odds + gumbel, k=budget).indices
                chosen_u, chosen_v = u[selected], v[selected]
                projected[item, chosen_u, chosen_v] = 1
                projected[item, chosen_v, chosen_u] = 1
        if not torch.equal(projected, projected.transpose(1, 2)):
            raise AssertionError("budget projection produced an asymmetric adjacency")
        if not torch.equal(
            projected[~batch["update_mask"]],
            batch["clean_adjacency"][~batch["update_mask"]],
        ):
            raise AssertionError("budget projection changed frozen context")
        return projected
