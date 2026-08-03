"""
Full Concept-based Similarity Reasoning network (inference logic, Sec. 2.1)
plus the doctor-in-the-loop test-time interactions (Sec. 3.2 / 3.3).

Pipeline for an input x:
  f            = F(x)                              (B, C, H, W)
  f'(h,w)      = P(f(h,w))                         (B, HW, C')     Eqn 10 (proj)
  S_km(h,w)    = <p_km, f'(h,w)>                   (B, K, M, H, W) cosine sim maps
  s_km         = max_{h,w} S_km(h,w)               (B, K*M)        Eqn 10 (max)
  logits       = H(s)                              (B, num_classes) Eqn 1
"""
import torch
import torch.nn as nn


class TaskHead(nn.Module):
    """
    H: a single linear layer mapping the K*M similarity scores to classes (Eqn 1).

    Input standardisation
    ---------------------
    The similarity scores live in a narrow band (empirically mean ~0.09, std
    ~0.06, with class-discriminative differences of ~0.005). Feeding those raw
    into a Linear layer under AdamW + weight decay gives a vanishing gradient
    signal, and the head degenerates to predicting the majority class. We
    therefore standardise the input with a BatchNorm1d.

    This does NOT break the paper's interpretability requirement: at inference
    BatchNorm is a fixed per-feature affine map, so `Linear(BN(s))` is still
    exactly a linear function of s. Call `effective_linear()` to fold the two
    into the single equivalent (W, b) whenever you need to reason about the
    contribution of each prototype.
    """
    def __init__(self, in_features=400, num_classes=3, normalize_input=True):
        super().__init__()
        self.normalize_input = normalize_input
        self.norm = nn.BatchNorm1d(in_features) if normalize_input else nn.Identity()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, s):
        return self.linear(self.norm(s))

    @torch.no_grad()
    def effective_linear(self):
        """Fold BN + Linear into a single equivalent linear map (W_eff, b_eff)."""
        W, b = self.linear.weight, self.linear.bias
        if not self.normalize_input:
            return W.clone(), b.clone()
        bn = self.norm
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)   # (F,)
        shift = bn.bias - bn.running_mean * scale                 # (F,)
        W_eff = W * scale.unsqueeze(0)
        b_eff = b + W @ shift
        return W_eff, b_eff


class CSRNetwork(nn.Module):
    def __init__(self, concept_model, projector, prototypes, task_head):
        super().__init__()
        self.concept_model = concept_model
        self.projector = projector
        self.prototypes = prototypes
        self.task_head = task_head

    # ------------------------------------------------------------- core maps
    def similarity_maps(self, x):
        """
        Returns:
          S_maps : (B, K, M, H, W)  cosine similarity maps for every prototype
          f       : (B, C, H, W)    backbone feature map (for local vectors etc.)
        """
        _, _, _, f = self.concept_model(x)
        B, C, H, W = f.shape
        patches = f.view(B, C, H * W).permute(0, 2, 1).reshape(B * H * W, C, 1)
        f_prime = self.projector(patches).view(B, H * W, -1)      # (B, HW, C')
        p = self.prototypes.get_normalized_prototypes()           # (K, M, C')
        K, M, Cp = p.shape
        S = torch.einsum("bnc,kmc->bnkm", f_prime, p)             # (B, HW, K, M)
        S_maps = S.permute(0, 2, 3, 1).reshape(B, K, M, H, W)
        return S_maps, f

    # ------------------------------------------------------------- inference
    def forward(self, x):
        S_maps, _ = self.similarity_maps(x)
        B, K, M, H, W = S_maps.shape
        s_scores = S_maps.reshape(B, K, M, H * W).max(dim=-1).values  # (B, K, M)  Eqn 10
        s_flat = s_scores.reshape(B, K * M)
        logits = self.task_head(s_flat)                              # (B, classes) Eqn 1
        return logits, s_flat, S_maps

    # --------------------------------------------------- test-time interaction
    @torch.no_grad()
    def predict_interactive(self, x, importance_map=None, rejected_concepts=None,
                            prototype_mask=None):
        """
        Sec. 3.3 test-time interaction.

        importance_map    : (H,W) or (B,H,W) tensor in [0,1] (A in Eqn 12). Draw
                            positive boxes as 1, negative boxes as 0, neutral=alpha.
        rejected_concepts : iterable of concept indices k to zero out (concept-level
                            interaction: sets s_km = 0 for all m in that concept).
        prototype_mask    : (K,M) {0,1} tensor to drop "unqualified" prototypes
                            (train-time atlas refinement, Sec. 3.2).

        Applies Eqn 13: clip negatives of S, multiply by A, then max-pool.
        """
        S_maps, _ = self.similarity_maps(x)
        B, K, M, H, W = S_maps.shape

        if importance_map is not None:
            A = importance_map.to(S_maps)
            if A.dim() == 2:
                A = A.view(1, 1, 1, H, W)
            elif A.dim() == 3:
                A = A.view(B, 1, 1, H, W)
            # S in [-1,1], A in [0,1]: clip negatives so weighting is monotone.
            S_maps = torch.clamp(S_maps, min=0.0) * A

        s_scores = S_maps.reshape(B, K, M, H * W).max(dim=-1).values  # (B, K, M)

        if prototype_mask is not None:
            s_scores = s_scores * prototype_mask.to(s_scores).view(1, K, M)
        if rejected_concepts:
            for k in rejected_concepts:
                s_scores[:, k, :] = 0.0

        logits = self.task_head(s_scores.reshape(B, K * M))
        return logits, s_scores, S_maps
