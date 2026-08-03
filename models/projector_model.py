"""
Phase-2 components: projector P, concept prototypes {p_km}, and the
multi-prototype contrastive loss (Eqns 5-9).

Supplementary Sec. 1: "The projector P includes 3 blocks of 1D convolution
layer, followed by an IBN and a ReLU activation with a residual connection.
The final output from P is L2-normalized." M = 100 prototypes per concept.
Supplementary Sec. 3: lambda = 20, gamma = 1000, delta = 0.1 for all datasets.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IBN1d(nn.Module):
    """
    Instance-Batch Normalisation for 1D concept vectors (IBN-Net, ECCV'18).

    Half the channels get Instance/vector normalisation, half get Batch
    normalisation. Concept vectors here have shape (N, C, 1) (a single spatial
    location), so InstanceNorm1d has no spatial extent to work with; in that
    degenerate case we normalise each sample across its instance-channels
    (GroupNorm with one group), which is the intended per-instance behaviour.
    """
    def __init__(self, channels, ratio=0.5):
        super().__init__()
        self.instance_channels = int(channels * ratio)
        self.batch_channels = channels - self.instance_channels
        self.instance_norm = nn.InstanceNorm1d(self.instance_channels, affine=True,
                                               track_running_stats=False)
        self.vector_norm = nn.GroupNorm(1, self.instance_channels)
        self.batch_norm = nn.BatchNorm1d(self.batch_channels)

    def forward(self, x):
        inst, batch = torch.split(x, [self.instance_channels, self.batch_channels], dim=1)
        if inst.size(-1) > 1:
            inst = self.instance_norm(inst)
        else:
            inst = self.vector_norm(inst)
        # BatchNorm needs >1 value per channel; guard the pathological case.
        vals_per_channel = batch.numel() // max(self.batch_channels, 1)
        if self.training and vals_per_channel <= 1:
            batch = F.batch_norm(batch, self.batch_norm.running_mean, self.batch_norm.running_var,
                                 self.batch_norm.weight, self.batch_norm.bias,
                                 training=False, eps=self.batch_norm.eps)
        else:
            batch = self.batch_norm(batch)
        return torch.cat([inst, batch], dim=1)


class Projector(nn.Module):
    """
    P: R^{in_dim} -> R^{out_dim}, applied per feature patch / concept vector.
    3 residual (Conv1d(k=1) + IBN + ReLU) blocks; output is L2-normalised so the
    dot product with (also normalised) prototypes is a cosine similarity.
    """
    def __init__(self, in_dim=768, out_dim=256):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, out_dim, kernel_size=1)
        self.block1 = nn.Sequential(nn.Conv1d(out_dim, out_dim, 1), IBN1d(out_dim), nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv1d(out_dim, out_dim, 1), IBN1d(out_dim), nn.ReLU())
        self.block3 = nn.Sequential(nn.Conv1d(out_dim, out_dim, 1), IBN1d(out_dim), nn.ReLU())

    def forward(self, x):
        # x: (N, in_dim, 1)  -> either a batch of concept vectors or feature patches.
        x = self.proj(x)
        x = x + self.block1(x)
        x = x + self.block2(x)
        x = x + self.block3(x)
        x = F.normalize(x, p=2, dim=1)
        return x.squeeze(-1)  # (N, out_dim)


class ConceptPrototypes(nn.Module):
    """
    M learnable prototypes per concept: shape (K, M, C').

    Init note: we deliberately use plain torch.randn (unit-variance Gaussian) and
    do NOT rescale with a small std. Prototypes are compared to L2-normalised
    vectors via cosine similarity, so what matters is their *directions* on the
    unit sphere. Small-std init (e.g. std=0.02, common for transformer weights)
    collapses all K*M prototypes into a tiny cluster in nearly the same direction,
    from which contrastive training cannot recover -- observed as ~95% dead
    prototypes when the atlas is built. Unit-variance randn spreads them roughly
    uniformly on the sphere in 256-D, which is the correct starting point.
    """
    def __init__(self, num_concepts=4, M=100, dim=256):
        super().__init__()
        self.num_concepts, self.M, self.dim = num_concepts, M, dim
        self.prototypes = nn.Parameter(torch.randn(num_concepts, M, dim))

    def get_normalized_prototypes(self):
        return F.normalize(self.prototypes, p=2, dim=-1)


class MultiPrototypeContrastiveLoss(nn.Module):
    """
    Multi-prototype contrastive objective (Eqns 6-9).

    For a projected concept vector v' with ground-truth concept k~:
      q_m   = softmax(gamma * <p_{k,m}, v'>)            over m         (Eqn 6)
      sim_k = sum_m q_m * <p_{k,m}, v'>                                (Eqn 7,8)
      loss  = -log  exp(lambda*(sim_k~ + delta))
                    ------------------------------------                (Eqn 9)
                    sum_k exp(lambda*(sim_k + delta*[k==k~]))
    The margin delta is added to the positive concept in both numerator and
    denominator, so plain cross-entropy over lambda*sim (with delta added to the
    positive logit) is exactly this objective.
    """
    def __init__(self, lambda_scale=20.0, gamma=1000.0, delta=0.1, class_weight=None,
                 diversity_beta=0.0, diversity_gamma=10.0):
        """
        class_weight: optional (K,) tensor, one weight per concept, passed straight
        through to F.cross_entropy. Phase 2 sees one training vector per
        (image, present-concept) pair, so the vector pool is exactly as imbalanced
        as concept prevalence (e.g. ISIC: streaks/negative_network are ~6% of all
        vectors each). Plain cross-entropy lets the projector+prototypes fit the
        dominant concepts and undertrain the rare ones -- weighting counteracts that,
        the same way Phase 1's BCE pos_weight does for the concept detector.

        diversity_beta: weight on a load-balancing penalty (see _diversity). Fixes
        the residual within-concept collapse that revival alone cannot: even with
        many alive prototypes, one prototype can still claim the majority of a
        concept's vectors (observed: pigment_network's top prototype held 71% of
        vectors, so a doctor rejecting it drops macro-F1 by 4.4). This penalty
        pushes the assignment to spread across prototypes so no single one is
        load-bearing, which is the precondition for safe doctor-in-the-loop
        rejection. 0.0 = off.
        """
        super().__init__()
        self.lam, self.gamma, self.delta = lambda_scale, gamma, delta
        self.diversity_beta, self.diversity_gamma = diversity_beta, diversity_gamma
        self.last_diversity = 0.0   # stashed for logging
        self.register_buffer("class_weight", class_weight, persistent=False)

    def forward(self, v_prime, labels, prototypes):
        # v_prime:(N,C')  labels:(N,)  prototypes:(K,M,C') normalised
        K, M, C = prototypes.shape
        cos = torch.einsum("nc,kmc->nkm", v_prime, prototypes)  # (N, K, M)
        q = F.softmax(self.gamma * cos, dim=-1)                 # (N, K, M)
        sim_k = (q * cos).sum(dim=-1)                           # (N, K)
        one_hot = F.one_hot(labels, num_classes=K).float()
        logits = self.lam * (sim_k + one_hot * self.delta)
        loss = F.cross_entropy(logits, labels, weight=self.class_weight)

        if self.diversity_beta > 0:
            div = self._diversity(cos, labels, K, M)
            self.last_diversity = float(div.detach())
            loss = loss + self.diversity_beta * div
        return loss

    def _diversity(self, cos, labels, K, M):
        """
        Switch-Transformer load-balancing auxiliary loss (Fedus et al. 2021),
        applied per concept.

        For concept k, restrict to the vectors whose TRUE concept is k -- those
        are the vectors being pulled toward k's prototypes; how they distribute
        over k's M prototypes is what determines whether any single prototype
        becomes load-bearing.

            loss_k = M * sum_m  f_m * P_m
              f_m = fraction of these vectors whose argmax prototype is m  (hard)
              P_m = mean soft-assignment probability to m                  (soft)

        = 1 at perfectly uniform routing, up to M when all vectors pile on one
        prototype. Averaged over concepts.

        TEMPERATURE: the main assignment softmax uses gamma=1000, which is
        saturated (one-hot) and carries ~0 gradient -- a penalty computed on it
        could not move anything. We therefore compute P_m with a much softer
        diversity_gamma so the term has real gradient to spread the prototypes.
        f_m uses the hard argmax and is detached, exactly as in the Switch loss.
        """
        total = cos.new_zeros(())
        n_active = 0
        for k in range(K):
            sel = labels == k
            nk = int(sel.sum())
            if nk == 0:
                continue
            cos_k = cos[sel, k, :]                                        # (nk, M)
            hard = cos_k.argmax(dim=-1)                                   # (nk,)
            f = torch.bincount(hard, minlength=M).float() / nk           # (M,)
            P = F.softmax(self.diversity_gamma * cos_k, dim=-1).mean(0)   # (M,)
            total = total + M * (f.detach() * P).sum()
            n_active += 1
        return total / max(n_active, 1)