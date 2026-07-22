# Scalable Graph Generation via Coarsening + Block Diffusion — Project Brainstorm

**Status:** internal brainstorm / related-work map, not a draft. Written to share with a collaborator.
**Goal of the project:** generate large sparse graphs (target: planar graphs with tens of thousands of nodes, ~40k) by (1) coarsening the training graphs with a high-quality Loukas instance, (2) running a classical discrete diffusion in the coarse latent space, and (3) decoding back to full resolution with a *factorized block diffusion* that exploits the sparsity of the coarse graph. The decoder is the heart of the contribution.

---

## 0. One-paragraph summary of the idea

We build a collection of pairs $(G, G_c)$ with spectrum-preserving Loukas coarsening (our own re-implementation, see Section 7). A standard DiGress-style discrete diffusion models the coarse graph $p_\theta(G_c)$. To decode, a size predictor tells us how many original nodes each supernode represents (for padding), then a shared discrete diffusion generates each intra-cluster subgraph, and finally inter-cluster (bipartite) edges are generated **only for pairs $(i,j)$ that are edges in $G_c$**. The key design question is how to make the inter-cluster generation both parallel and globally coherent; our proposal is a multi-pass, variable-grouping refinement scheme rather than strict autoregression.

---

## 1. The pipeline, stage by stage, with provenance

Notation: $G=(A,X)$ original graph, $N$ nodes. $G_c=(A_c,X_c)$ coarse graph, $n_c$ supernodes. $Q\in\{0,1\}^{N\times n_c}$ lifting (one 1 per row), $P\in\mathbb{R}^{n_c\times N}$ reduction, $X_c=PX$. $E_c$ = edge set of $G_c$. $B_i$ = set of original nodes in cluster $i$; $v_i=|B_i|$. $\bar b$ = mean cluster size.

| Stage | What it does | Closest prior art | What is ours |
|---|---|---|---|
| **(A) Coarsening / build $(G,G_c)$ collection** | spectrum-preserving Loukas, produce many coarse graphs | Loukas & Vandergheynst 2018 (REC); used by ETH-2024 and LGDC | our Loukas re-implementation (normalized-Laplacian-preserving, more equitable cluster sizes, bigger diversity of coarse forests — Section 7) |
| **(B) Latent diffusion on $G_c$** | discrete diffusion (DiGress-style) over the small coarse graph | DiGress (Vignac 2023); same as LGDC stage 1 | nothing novel here on purpose; this part is deliberately standard |
| **(C) Cluster-size prediction** | predict $v_i$ = #original nodes per supernode, to pad the intra diffusion correctly | PARD's block-size predictor (Algo. 2, cross-entropy); HiGen uses parent edge weights | a small GCN / basic graph transformer whose *only* job is $v_i$; not the generator |
| **(D) Intra-cluster generation** | diffusion that expands each supernode into its subgraph $B_i$ | ETH-2024 expansion; HiGen community generation; PARD intra-block diffusion | parallel over all clusters, conditionally independent given $G_c$ |
| **(E) Inter-cluster (bipartite) generation** | generate cross-edges for each $(i,j)\in E_c$ | HiGen bipartite prediction; ETH-2024 candidate inter-cluster edges | **the main contribution**: sparsity-structured, multi-pass, variable-grouping refinement (Section 4) |

People to name in the write-up:
- **Loukas & Vandergheynst (ICML 2018)** — restricted spectral approximation (RSA), the coarsening guarantee everyone builds on.
- **Vignac et al., DiGress (ICLR 2023)** — discrete denoising diffusion for graphs; the latent-diffusion backbone. DiGress saturates around 100–200 nodes (their own README points large-graph users to SparseDiff).
- **Qin/Frossard group, SparseDiff (arXiv 2023, "Sparse Training of Discrete Diffusion Models")** — sparse version of DiGress; see Section 5.
- **Bergmeister, Martinkus, Perraudin, Wattenhofer, ETH (ICLR 2024)** — "Efficient and Scalable Graph Generation through Iterative Local Expansion"; coarsening-as-forward-diffusion, expand+refine reverse. Record affiched: point-cloud graphs up to 5037 nodes.
- **Karami, HiGen (ICLR 2024)** — hierarchical community-parallel + bipartite cross-edge generation. **Our closest competitor — Section 3.**
- **Zhao, Ding, Akoglu, PARD (NeurIPS 2024)** — permutation-invariant autoregressive *diffusion*, block-by-block via a structural partial order. **Section 2.**
- **Arriola et al., BD3-LM (ICLR 2025 oral)** — block diffusion for *text*: diffusion inside blocks, autoregressive across blocks. The conceptual frame. **Section 2.**
- **Ma et al., "Accelerating Discrete Diffusion Decoding with Parallel Scan" (ICLR 2026 submission)** — *associative states*. The justification for our multi-pass parallel refinement. **Section 6.**
- **Osman, Jiang, Buffelli, Dong, Toni, LGDC (NeurIPS 2025 workshop)** — latent graph diffusion via spectrum-preserving coarsening. Closest in framing, weakest in novelty — Section 3.

---

## 2. Block diffusion: BD3-LM and PARD (the two reference points)

**BD3-LM (text).** Split a token sequence into $K$ non-overlapping blocks. Run discrete diffusion *within* each block; factorize *across* blocks autoregressively, $\log p(x)=\sum_k \log p(x^k\mid x^{<k})$. Tuning block size interpolates AR (block=1) and full diffusion (block=all). Contributions worth knowing: KV-caching, arbitrary-length generation, and an analysis of the *gradient variance* of the diffusion objective with noise schedules that reduce it. For us: BD3-LM is the conceptual parent ("diffusion inside blocks, something across blocks"), but it is text and strictly autoregressive across blocks. We cite it as the frame, not the method.

**PARD (graphs).** This is block diffusion already instantiated on graphs.
- Splits a graph into blocks via a **structural partial order** $\phi$ (Algo. 1): a permutation-equivariant hashing of multi-hop degree profiles; nodes with the same structural color share a rank. Blocks are added in order, conditioned on previous blocks: $p_\theta(G)=\prod_i p_\theta(G[B_{1:i}]\setminus G[B_{1:i-1}] \mid G[B_{1:i-1}])$.
- Each block's conditional is a **shared discrete diffusion** with an equivariant network (PPGN-Transformer hybrid, 3-WL).
- A separate **block-size predictor** (their Algo. 2) predicts $|B_{i+1}|$ — this is exactly the role of our Stage (C).
- **Parallel training only**: a GPT-style causal mask (their Eq. 10) lets all $K_B$ conditionals be computed in one pass at *training time*. **At generation time PARD is strictly sequential** (Algo. 4): block $i$ needs clean $G[B_{1:i-1}]$ before denoising block $i$. They measure ~94.7% of generated paths match the training path (5% exposure-bias divergence).

**Theoretically important result from PARD for us (Thm 3.2 / Prop 3.3):** no permutation-equivariant network, however expressive, can perform a general graph transformation without symmetry breaking; structurally equivalent nodes/edges get identical representations. The cure is injecting noise (diffusion). *Consequence for our pipeline:* our intra-cluster expansion **must** be a diffusion, not a deterministic GCN, because a cluster may unfold into structurally-equivalent nodes that a deterministic equivariant decoder cannot distinguish. This is a feature: it justifies why Stage (D) is diffusion.

**How we differ from PARD (state this clearly, it is the crux):**
- PARD's blocks come from a *structural partial order on the original graph*; ours come from *coarsening clusters*. Number of PARD blocks grows with graph size and they are added one-by-one.
- PARD is autoregressive across blocks → sequential decoding, can stall/diverge (exposure bias), not parallelizable at inference.
- In our scheme the coarse graph $G_c$ is an explicit *conditioning scaffold* that makes the intra-cluster subgraphs **conditionally independent given $G_c$**, so Stage (D) is one parallel batch, not a chain. PARD cannot do this because its blocks carry autoregressive dependencies.

Framing line for the paper: *the coarse graph provides a conditioning structure that converts an autoregressive chain of length $K_B$ into two mostly-parallel passes (intra, then inter).*

---

## 3. The two real competitors: HiGen and LGDC

### 3.1 HiGen (Karami, ICLR 2024) — closest, take seriously

HiGen already does "communities in parallel, then cross-edges (bipartites) with a separate network", coarse-to-fine. Concretely:
- Recursively partitions $G$ into a hierarchy $\mathcal{HG}=\{G^0,\dots,G^L\}$; each community = diagonal block of the adjacency, each bipartite $\mathcal{B}_{ij}$ = off-diagonal block. Super-node weight = sum of community edge weights; super-edge weight = sum of bipartite edge weights (so coarse edges are integer-weighted).
- Factorizes $p(G^l\mid G^{l-1})=\prod_i p(\mathcal{C}^l_i\mid G^{l-1})\prod_{(i,j)\in E(G^{l-1})} p(\mathcal{B}^l_{ij}\mid G^{l-1},\{\mathcal{C}^l_k\})$. **Communities conditionally independent given the parent; bipartites conditionally independent of each other given parent + communities.** (Their Thm 3.1, via a multinomial stick-breaking factorization.)
- Each component is modeled as a **multinomial over edge weights**, generated autoregressively within the community; the network is GNN-based (not diffusion).

**This is uncomfortably close to our Stages (D)+(E).** We must name HiGen as the primary baseline and articulate the differences crisply. Honest differences:
1. **Generative engine.** HiGen = multinomial / autoregressive weighted-edge model. Ours = discrete diffusion inside blocks. PARD's Thm 3.2 argues diffusion is needed to break symmetries that an equivariant non-diffusion model cannot; this is a principled reason to expect diffusion to beat HiGen's multinomial on symmetry-rich graphs (grids, planar).
2. **Where the partition comes from.** HiGen uses generic graph clustering (Louvain-style community detection) chosen to maximize modularity. We use **spectrum-preserving Loukas with an RSA guarantee** — the partition is chosen to preserve the Laplacian spectrum, not just community density. This is what unlocks the theory in Section 8 (RSA $\varepsilon$ → generation fidelity), which HiGen cannot state.
3. **Inter-cluster coherence.** HiGen assumes bipartites are *mutually independent given the parent + communities* — a hard independence assumption. **We do not want to inherit it blindly**; our multi-pass variable-grouping scheme (Section 4) is designed precisely to *relax* this assumption and measure the cost of it. This is a concrete, defensible delta.
4. **Scale.** HiGen targets moderate graphs. Our headline claim is order-of-magnitude larger (Section 5).

Risk to watch: a reviewer who knows HiGen will say "communities-in-parallel + separate bipartite net is HiGen." The answer must be: *diffusion engine (principled via PARD Thm 3.2) + spectral coarsening with fidelity theory + a refinement scheme that removes HiGen's bipartite-independence assumption.* If we cannot beat HiGen empirically on shared benchmarks, the paper is in trouble — so HiGen must be in every table.

### 3.2 LGDC (NeurIPS 2025 workshop) — closest in framing, weak novelty

LGDC = Loukas spectrum-preserving coarsening + DiGress diffusion on the coarse graph + ETH-2024's expand+refine, but with a **single** coarse→fine pass instead of ETH's multiple cycles. That single-step change is essentially their whole delta over ETH-2024; their own related work admits the framing is Bergmeister et al. with one expansion step. Complexity they quote: $O(n^2 + Tn_c^2)$.

**Why we are not "LGDC + PARD":** LGDC keeps ETH's expand+refine, which puts the full candidate structure into one (dense) graph-transformer pass — exactly the $O(n^2)$ bottleneck. **We replace the entire expansion mechanism** with a factorized block diffusion (intra parallel + inter only on $E_c$) plus a multi-pass refinement. The decoder is a different object, and it is the main contribution. By contrast LGDC's modification of ETH is a one-line change. Our delta on the decoder is strictly larger than LGDC's delta on ETH.

---

## 4. Stage (E), the hard part: generating inter-cluster edges

This is where the design effort is, and where the open questions live.

### 4.1 The problem
After intra-cluster subgraphs are generated, we must create cross-edges for each $(i,j)\in E_c$. The sparsity argument: if $(i,j)\notin E_c$ then clusters $i$ and $j$ had no edge between them in the original graph (by construction of the coarsening — Loukas contracts, so $A_c$ aggregates inter-cluster edges). So we only instantiate $|E_c|$ bipartite problems, **not** $n_c^2$. This is the source of scalability.

The worry (stated by us, must be addressed): generating the bipartite for $(i,j)$ changes the context that should have conditioned $(i,k)$. Independently generated bridges can be globally incoherent — the classic failure mode of subgraph-aggregation methods (SaGess; and HiGen's bipartite-independence assumption is exactly this risk, made into an assumption).

### 4.2 Candidate designs (these are the options to discuss / ablate)

**Option 0 — shared denoiser on the induced pair-subgraph (recommended v1).** Do **not** build a special bipartite architecture. For pair $(i,j)$, instantiate the *same* discrete denoiser used for intra, but on the induced subgraph $B_i\cup B_j$, with intra edges already fixed (frozen) and only the cross-edges to be denoised (masked). One shared network handles both "single cluster" (intra) and "cluster pair" (inter). Simplest, reuses weights, smallest code. Per-pair cost $O((v_i+v_j)^2)$, dominated overall by $|E_c|\cdot\bar b^2$.

**Option 1 — conditional independence given $(G_c$ + intra subgraphs$)$.** Assume each bipartite $(i,j)$ depends only on $G_c$ (carried via a message-passing pass over the coarse graph) plus the two already-generated intra subgraphs. Then all bipartites are conditionally independent and generated in **one parallel batch**. This is the maximally-parallel design and the cleanest claim. It is essentially HiGen's assumption but with a diffusion engine and richer conditioning. The empirical question: how much fidelity is lost? Measure it.

**Option 2 — multi-pass variable-grouping refinement (our preferred contribution).** This is the Cluster-GCN / GraphSAINT intuition transported to generation. Generate all bipartites in parallel (pass 1, e.g. via Option 1). Then run additional passes in which we **re-group clusters differently** and re-denoise the cross-edges conditioned on the current state of neighbors. Because the grouping/order changes between passes, no inter-cluster boundary is permanently frozen: an edge that was a "boundary" (cut) in one pass becomes "interior" in another, letting information flow across it. Concretely we can vary which blocks are merged at different diffusion steps — sometimes denoise pair $(i,j)$ alone, sometimes the triple $\{i,j,k\}$ together, etc. — analogous to Cluster-GCN re-sampling different clusters each epoch so that a node sees different macro-neighborhoods. Doing this *across diffusion steps* should improve coherence without paying the full sequential cost.
  - Pro: parallel, no strict ordering, directly attacks the bipartite-interference problem, removes HiGen's independence assumption.
  - Con: must argue/measure that the passes **converge** to a coherent joint distribution. A Gibbs-sampling / fixed-point-iteration framing helps: each pass is a conditional re-sampling of cross-edges given neighbors; under mild conditions repeated conditional re-sampling targets the joint. This is also where the *associative states* paper (Section 6) gives cover: independently sample, then refine once context is available.

**Option 3 — light autoregression on the coarse graph.** Impose a partial order on $E_c$ (PARD-style but at the coarse level) and generate bipartites semi-autoregressively. Since $G_c$ is small ($n_c\ll N$), this is cheap. Safest fallback if Options 1/2 lose too much fidelity. But it reintroduces sequential dependence and the exposure-bias risk, so keep it as a baseline/ablation, not the headline.

**Recommendation.** v1 = Option 0 for the engine + Option 1 for parallelism, to get something running and measure the incoherence. Then introduce Option 2 as the contribution and show it closes the gap to (or beats) Option 3 while staying parallel. Report all three in an ablation: pure-independent (1), multi-pass refinement (2), coarse-AR (3).

### 4.2bis The global-degree problem (the real weakness of Option 0)

Independent per-pair generation respects a *local* constraint per bipartite but the binding constraint is *global per node* (summed over all of a node's bipartites). Molecular case: a carbon (valence 4) gets 2 cross-edges from bipartite $(i,j)$; the separate call for $(i,k)$, if it does not know those 2 are already spent, can add more and break valence. This is exactly why **HiGen sidesteps it by assumption** (bipartites mutually independent given the parent) — fine for graphs where degree is soft, broken for molecules where valence is hard — and why **PARD does not have the problem**: each block sees the whole graph generated so far, so consumed degree is always visible.

Fix (state, do not over-prescribe): expose **consumed degree** to every inter call. Each border node carries, in its features, how much degree it has already used (intra + previously-generated bipartites), so the denoiser sees the *remaining budget*, not a "fresh" node. Note this re-introduces an ordering dependency between bipartites of the same node, so it pushes Option 0 toward Option 2/3.

The parallelism-preserving version: predict each node's **total target degree up front** (same spirit as PARD's degree/block-size prediction, Stage C) and feed it as a feature from the start; then each bipartite knows "must end at degree $d$, has used $u$, $d-u$ remain" without needing to see the other bipartites in detail. The predicted target degree becomes the coordination channel that replaces direct bipartite-to-bipartite communication. Pass 1 generates all bipartites in parallel against this budget; a refinement pass (Option 2) corrects over-saturated nodes.

**Important design preference (do not inject chemical priors):** we condition on a *graph* quantity (target/consumed degree), not on chemistry. We do **not** hand the model the valence rules — the model must learn the degree budget from the atom type itself. Degree is a structural feature the model would see anyway; valence-from-atom-type is something it should infer, not be told.

**Hard-masking as a safety net (à la DiGress):** DiGress applies a validity mask at sampling time (forbid an edge that would exceed valence). We can do the same as a *secondary* guard on top of the degree conditioning. But masking alone is insufficient: if the model never saw the budget, it proposes systematically invalid configs and masking yields artifacts. Conditioning is primary, masking is the net.

### 4.3 Architecture question: dense transformer vs sparse attention for the pair comparison
We do **not** put all of $A$ (or even all of $A_c$) into a dense transformer — that would reproduce the exact LGDC/ETH bottleneck we are trying to beat. Sparsity is handled by **which pairs we instantiate** ($E_c$), not by an attention mask inside a giant transformer. Each instantiated problem (a cluster, or a cluster pair) is *small* ($v_i$ or $v_i+v_j$ nodes), so a dense PPGN / PPGN-Transformer denoiser *locally* is fine and gives the 3-WL expressivity we want for local motifs. So: **no global sparse attention needed**; we factorize into many small shared denoisers rather than one big masked one. This is closer to PARD's shared-network-per-block than to SparseDiff's single global masked model, and it is the cleaner scalability story.

### 4.4 Caching
PARD mentions caching as future work; BD3-LM uses KV-caching across blocks. For us, two caching opportunities: (a) cache the coarse-graph message-passing embedding of $G_c$ once and reuse it as conditioning for every intra and inter problem; (b) in the multi-pass scheme, cache frozen-neighbor representations between passes so only the re-grouped region is recomputed. Worth a paragraph; not load-bearing for the first version.

### 4.5 Is bipartite diffusion a thing? (state of the art)
Honest answer: there is **no established SOTA for generating only the edges between two fixed node sets via diffusion**. It is essentially generative bipartite link prediction. HiGen and ETH-2024 do it implicitly inside a larger model; nobody isolates it as a dedicated diffusion module. Two readings:
- It is a genuine **gap** we could fill (a clean "conditional bipartite edge diffusion" module would be a small but real contribution).
- Or it is a sign we **don't need** a special module — Option 0 (shared denoiser on $B_i\cup B_j$ with masking) already covers it.
Decision: use Option 0 for v1; keep "dedicated bipartite diffusion" as an extension only if bridges underperform.

---

## 5. SparseDiff and the alternative framing (this is a strong angle)

**SparseDiff (Qin/Frossard group, "Sparse Training of Discrete Diffusion Models for Graph Generation").** Key idea: real large graphs are sparse, so instead of predicting all $N^2$ pairs, (a) use a noising trajectory that **preserves sparsity** (the noisy graph stays sparse instead of becoming dense mid-diffusion — the main memory killer of DiGress), and (b) at each step predict only a **subset of edges**, making space complexity linear in the number of chosen edges; at inference, progressively fill the adjacency by subsets. It is *not* the original DiGress group's flagship but a scaling-focused follow-up. Scale reached: planar 64, SBM up to 200, Ego/Protein up to 500, one example at ~1045 nodes. So SparseDiff buys **both** speed and somewhat larger graphs, but stays around ~1000 nodes and keeps a **single global model** (it plays on the edge-subset mask, not on factorization).

**Our alternative framing (worth considering as the main pitch):** SparseDiff selects the edge subset *at random*. We can argue the coarse graph gives the **structurally optimal prior** on which edges to predict: only instantiate cross-edges where $A_c$ has an edge. So instead of "random subset + many uninformed iterations", we have "structurally-informed subset from the coarsening, no uninformed iterations." This positions SparseDiff as a direct baseline we dominate on a clear axis. Two ways to use this:
- As the *main* framing: "the coarse graph is the right prior for sparse edge prediction."
- Or as one strong section. Either way, do not undersell it — it gives a crisp comparison and a clear win condition.

**Scale table to internalize (for positioning):**

| Method | Largest graphs reported | Engine | Single model vs factorized |
|---|---|---|---|
| DiGress | ~64 planar / ~200 SBM (README: use SparseDiff beyond 100–200) | discrete diffusion | single |
| SparseDiff | ~500 (Protein), 1045-node example | sparse discrete diffusion | single, random edge subset |
| HiGen | moderate | multinomial / AR | factorized (communities + bipartites) |
| ETH-2024 | **5037** (point cloud) | coarsening + expand/refine diffusion | iterative, dense refine |
| LGDC | small benchmarks (Planar, Tree, Comm-20) | coarsening + diffusion + single expand | iterative |
| **Ours (target)** | **~40000 planar** | coarsening + factorized block diffusion | factorized, $E_c$-structured |

Target ~40k is ~8x the ETH record (5037) and ~40x SparseDiff. **This number, if achieved with maintained quality, is the paper.** Everything else (architecture, theory) is supporting. Put the scaling result at the center.

---

## 6. The "associative states" paper — justification for multi-pass refinement

Ma et al., "Accelerating Discrete Diffusion Decoding with Parallel Scan" (ICLR 2026 submission, text, training-free). They observe that in block diffusion, a class of blocks — **associative states** — can be **sampled independently without conditioning on the prefix, then refined once the prefix becomes available** (a form of self-refinement). They build a parallel-scan decoder with *local remasking* + *global aggregation*, getting strong throughput vs strict semi-AR.

Why it matters to us: this is *exactly* the theoretical cover for Option 2. Our claim "generate bipartites in parallel, then refine when neighboring context is available" is the graph instantiation of associative-state parallel-scan decoding. Differences to stress: theirs is text and a 1-D sequence with arbitrary block cuts; ours is a graph where the **grouping comes from the coarse structure $E_c$**, not an arbitrary sequence split. So we inherit the legitimacy of "independent-then-refine beats strict AR in parallelism without quality loss" while contributing the structured (graph/coarsening-driven) version.

---

## 7. Our coarsening instance (to send to collaborator)

We re-implemented Loukas variation-neighborhood coarsening to **preserve the normalized Laplacian** (not just combinatorial), which gives (claimed, to verify together): (a) a larger / more diverse forest of coarse graphs $G_c$ in the training collection, and (b) **more equitable cluster sizes** $v_i$. Both matter for this project specifically:
- More equitable $v_i$ → the padding for the intra diffusion (Stage D) is less wasteful, and the size-predictor (Stage C) has an easier, lower-variance target. PARD notes the block-size prediction is important for final quality (they even condition diffusion on predicted degree); a better-behaved $v_i$ distribution directly helps.
- More diverse coarse forest → richer training collection for the latent diffusion (Stage B), analogous to ETH-2024 randomizing coarsening sequences during training (vs HiGen/Davies which pre-fix a single Louvain clustering). ETH explicitly argues random coarsening sequences help generalization; our diverse instance is in that spirit.
- Normalized-Laplacian preservation ties directly into the RSA fidelity theory below (Section 8): the cleaner the spectral preservation, the tighter the generation-fidelity bound.

(Action item: package this Loukas instance with a short README on what it preserves and how cluster-size equity is measured, so the collaborator can drop it into the pipeline.)

**On training-data volume.** The shared intra/inter denoiser is *not* trained on whole 40k-node graphs; it is trained on the **sub-problems**: every cluster $B_i$ and every pair $B_i\cup B_j$ of a training graph is one training example. A single 40k graph coarsened into ~10-node clusters already yields ~4000 intra examples + $|E_c|$ inter examples — so even one large graph fragments into thousands of sub-problems (same logic as SaGess sampling subgraphs, and Cluster-GCN). The denoiser learns the *local* distribution of clusters and bridges; the global structure comes from the latent diffusion on $G_c$. The tighter budget is the latent diffusion, which needs enough *distinct whole* $G_c$ — but for planar (and other synthetic targets) we can generate as many graphs as we want, and our diverse coarsening multiplies $G_c$ per graph further.

---

## 8. The theoretical cherry: RSA $\varepsilon$ → generation fidelity

This is what would move the paper from "good NeurIPS" to "very strong", and it is the thing **no competitor can write** (LGDC uses the spectral condition only as motivation, never as a bound; HiGen has no spectral guarantee; PARD has no coarsening). It is the cherry — to be developed once experiments work, not a blocker.

RSA constant of the coarsening: $\varepsilon=\sup_{x\in\mathcal{R},\,\|x\|_L=1}\|x-QPx\|_L$, where $\mathcal{R}$ is the signal subspace of interest (e.g. span of the $K$ smallest Laplacian eigenvectors). $\varepsilon=0$ means perfect coarsening.

Target decomposition of the total generation error:
$$
d(p_{\text{gen}}, p_{\text{data}}) \;\le\; \underbrace{d\big(p_\theta(G_c),\,p(G_c)\big)}_{\text{latent diffusion error}} \;+\; \underbrace{\mathbb{E}\big[d\big(p_\psi(G\mid G_c),\,p(G\mid G_c)\big)\big]}_{\text{expansion / decoder error}} \;+\; \underbrace{\text{term}(\varepsilon)}_{\text{coarsening distortion}}.
$$
The interesting object is the third term: even with a perfect latent diffusion and a perfect decoder, generation cannot beat what the coarsening preserves. If we turn LGDC's spectral inequality ($(1-\varepsilon)\mathrm{Tr}(X^\top LX)\le \mathrm{Tr}(X_c^\top L_c X_c)\le(1+\varepsilon)\mathrm{Tr}(X^\top LX)$) into a bound on a generation-relevant distance (e.g. spectral MMD between generated and data graphs), we have a real theorem.

Stretch goal connecting to our taxonomy work: the choice of reduction matrix $P$ (within the admissible sets $E_3\subseteq E_2\subseteq E_1$) changes $\varepsilon$ and therefore the fidelity bound. Showing that the $P$-choice has a *provable* effect on generative quality is something only our group can do, and it ties this paper to the taxonomy paper. This is the differentiator of last resort against any "= HiGen / = LGDC" reviewer.

The complexity result ($O(|E_c|\bar b^2 + \sum_i v_i^2)$, i.e. linear in $|E_c|$ hence in original sparsity) is trivial and we know it — include it, do not oversell it. The RSA bound is the non-trivial theory.

---

## 9. Open questions for the collaborator

1. **Convergence of the multi-pass refinement (Option 2).** Can we frame it as Gibbs / fixed-point iteration over cross-edges and get a convergence statement, or at least an empirical convergence diagnostic? This is the riskiest design choice.
2. **Bipartite independence cost.** How much fidelity does Option 1 (full conditional independence given $G_c$ + intra) actually lose vs Option 2/3 on planar / SBM? This number decides the whole decoder design.
3. **Beating HiGen.** On shared benchmarks (SBM, Protein, Ego, planar), can the diffusion engine + spectral coarsening beat HiGen's multinomial-AR? If not, what is the failure mode (bridges? cluster sizes? expressivity)?
4. **Size predictor calibration.** Is the GCN size predictor (Stage C) accurate enough that padding errors do not propagate? Does conditioning the intra diffusion on predicted $v_i$ (PARD-style degree conditioning) help?
5. **The 40k target.** What is the actual memory/time wall, and where does it bind first — coarse diffusion, intra batch, or the number of refinement passes?
6. **RSA bound.** Is the third error term ($\text{term}(\varepsilon)$) actually boundable for a generation-relevant metric, or only for the energy $\mathrm{Tr}(X^\top LX)$? Which $P$ (in $E_2$ vs $E_3$) gives the best bound vs the best empirical fidelity — do they agree?

---

## 10. Risk summary (honest)

- **Biggest external risk:** HiGen. "Communities in parallel + separate bipartite net, coarse-to-fine" is its abstract almost verbatim. We must (a) beat it empirically and (b) differentiate by engine (diffusion, justified by PARD Thm 3.2), partition (spectral/RSA), and the refinement scheme that drops its independence assumption.
- **Secondary risk:** being read as "LGDC + PARD." Counter: our decoder is a different mechanism (factorized block diffusion on $E_c$), and LGDC's own delta over ETH is one step. The decoder is the contribution, not the coarsening-then-diffuse skeleton.
- **What makes it clearly publishable:** the scaling number (~40k, vs 5037 ETH record), with HiGen and SparseDiff in every table, plus the RSA-fidelity theory as the closer.
- **What is genuinely novel (defensible):** (i) inter-cluster edge subset chosen by $E_c$ rather than at random (vs SparseDiff); (ii) multi-pass variable-grouping refinement that removes the bipartite-independence assumption (vs HiGen) and is parallel (vs PARD); (iii) RSA $\varepsilon$ → generation-fidelity bound tied to the $P$-taxonomy (vs everyone).
