# Evaluation Metrics — CSR on ISIC-2017

Every metric used anywhere in this project, what it means, how it is computed,
where it lives in the code, and its current value. Grouped by what it measures.

The dataset throughout is ISIC-2017: **4 concepts** (milia-like cyst, pigment
network, negative network, streaks) and **3 disease classes** (seborrheic
keratosis, melanoma, nevus). Splits: 2000 train / 150 validation / 600 test
(official ISIC-2017 Test v2). Class counts are heavily imbalanced
(train 254 / 374 / 1372; test 90 / 117 / 393), which is why macro-averaging is
used everywhere.

---

## A. Diagnostic performance (main task: disease classification)

These measure how well the final model predicts the disease. Reported by
`evaluate.py`.

### 1. Macro F1-score  ⟵ PRIMARY METRIC
The mean of the per-class F1 scores, treating all 3 classes equally regardless
of how many test images each has.

```
F1_class = 2 · precision · recall / (precision + recall)
Macro-F1 = (F1_seb + F1_mel + F1_nevus) / 3
```

This is the paper's headline metric (Table 1), chosen precisely because of the
class imbalance: plain accuracy would be dominated by the majority class
(nevus). A model that only ever predicted "nevus" would score 65% accuracy but a
terrible macro-F1.

- **Paper (CSR, ISIC):** 71.5
- **Our best static model** (`checkpoints_revive`): **65.66**
- **Concept-dropout model** (`checkpoints_dropout`): 64.28 clean, 64.94 with a
  correct doctor intervention (see Section E).

### 2. Per-class F1
The F1 of each disease individually. Exposes *where* the model struggles — for us
that is **melanoma** (F1 ≈ 47–54), the clinically hardest and a
minority class, versus nevus (F1 ≈ 81).

### 3. Accuracy
Fraction of test images classified correctly. Reported as a secondary number
only; under this imbalance it flatters the model (our 65.66 macro-F1 model has
~71% accuracy). Never used for model selection.

### 4. Precision & Recall (per class)
From `sklearn.classification_report`. Precision = of the images the model called
class X, how many really were X. Recall = of the true class-X images, how many
the model caught. F1 is their harmonic mean.

### 5. Confusion matrix
3×3 table, rows = true class, cols = predicted. Shows the *pattern* of mistakes
(e.g. melanoma being misread as nevus), not just the rate.

---

## B. Concept detection (Phase 1)

### 6. Concept F1 (`val_conceptF1`)
Phase 1 trains a multi-label detector for the 4 concepts (each present/absent).
A concept is predicted "present" when its sigmoid output > 0.5; concept-F1 is the
macro-F1 over the 4 concepts on the validation set.

- Used for **Phase-1 model selection** (best checkpoint = highest val_conceptF1).
- Peak achieved: **0.6827** (with class-weighted BCE). This ceiling matters —
  weak concept detection caps everything downstream.
- Code: `train_phase1.py`.

---

## C. Prototype learning (Phase 2)

### 7. Prototype assignment accuracy (`val_protoAcc`)
Fraction of local concept vectors whose nearest prototype belongs to the
*correct* concept. Used for Phase-2 model selection. Code: `train_phase2.py`.

### 8. Alive prototype count
Diagnostic (not a quality score): how many of the 400 prototypes receive at least
one assignment. Revealed **prototype collapse** — only 23–42 of 400 were ever
used — and confirmed the fix: random-restart revival raised it to **289–316 of
400**. Code: `train_phase2.py` (`revive_dead_prototypes`).

### 9. Top-prototype share (diversity diagnostic)
Per concept, the fraction of assignments captured by its single most-used
prototype. Used to check the load-balancing/diversity penalty (reduced
pigment_network's top share from 71% → 11.8%).

---

## D. Interpretability & trustworthiness

### 10. Pointing Game (PG) hit rate
For every (image, active-concept) pair, take the concept's best-prototype
similarity map, upsample to image resolution, and check whether its **argmax
falls inside that concept's ground-truth region** (reconstructed from the
official superpixel PNG + Part-2 feature JSON). The hit rate is the fraction that
land inside.

Measures whether the model "looks" at the right place — i.e. whether an
explanation is *trustworthy*, not just *present*. Paper Table 2: CSR 60.9%,
refined 79.5%. Requires the full-res Data zip (has superpixels). Code:
`pointing_game.py`.

### 11. Explanation size (`#exp`)
Number of explanations shown per prediction = **4** (one exemplar per concept).
Smaller = more interpretable. This is CSR's real edge in Table 1 (others use
90–600). Printed by `evaluate.py`.

### 12. Number of prototypes (`#pro`)
Total prototypes = **400** (4 concepts × 100). Printed by `evaluate.py`.

---

## E. Doctor-in-the-loop / interaction metrics (this project's additions)

These measure whether a doctor rejecting a concept *helps*. None of them exist in
the original paper for ISIC — they were built here to test the interaction claim.

### 13. Top1−top2 margin
The gap between the model's highest and second-highest class probabilities. Small
margin = the model is undecided. This is the **paper's own trigger**: intervene
only when margin < 0.3. Computed in `confidence_gate.py`, `selective_rejection.py`.

### 14. Confidence delta (ΔP_true)
Change in the predicted probability of the **correct** class after an
intervention. Positive = the intervention pushed the model toward the right
answer, even if the label didn't flip. Catches partial effects a flip-only metric
misses. Code: `confidence_delta_experiment.py`.

### 15. Fixed / Broke / Net
Under an intervention policy:
- **Fixed** = wrong → correct (good)
- **Broke** = correct → wrong (bad)
- **Net** = Fixed − Broke

The honest scoreboard for any doctor-in-the-loop policy. Used in
`confidence_gate.py`, `test_time_interaction.py`, `selective_rejection.py`.

### 16. Per-prototype ablation impact (Δmacro-F1)
For each of the 400 prototypes, zero its contribution in the task head and
measure the change in validation macro-F1. Produces a ranked "safety map" of how
much the classifier *relies* on each prototype. Code: `prototype_impact.py`.

### 17. Atlas hit_count vs. task-head reliance  ⟵ KEY DISTINCTION
- **hit_count**: how often a prototype wins the Phase-2 assignment competition.
- **task-head reliance**: how much the final classifier's weight on that
  prototype's score affects the prediction (metric 16).

These are **different, largely uncorrelated** statistics — a prototype can be
atlas-rare yet classifier-critical. This is a genuine gap in the paper's premise
that a doctor can safely review prototypes via the atlas. Demonstrated in
`prototype_impact.py` / `evaluate_curated.py`.

### 18. Over-firing
A concept's peak similarity on an image **minus** the level that concept normally
reaches when it is genuinely absent (learned from the training set). A large
positive value = the model is hallucinating the concept. Used as the (ground-
truth-free) trigger for *which* concept to reject. Code: `selective_rejection.py`.

### 19. Macro-F1 delta under an intervention policy
The net change in test macro-F1 when a whole intervention policy is applied.
The bottom-line number for "does doctor-in-the-loop help." Best result:
**+0.66** (64.28 → 64.94) on the concept-dropout model at the paper's own
margin < 0.3 (`selective_rejection.py`).

---

## Note: metrics vs. loss functions

Losses are what training *optimizes*; metrics are what we *report*. They are
deliberately different (we never select a checkpoint on raw loss — always on a
validation metric). The losses used:

| Phase | Loss | Optional class weighting |
|-------|------|--------------------------|
| 1 | `BCEWithLogitsLoss` (multi-label concepts) | `pos_weight` |
| 2 | `MultiPrototypeContrastiveLoss` (Eqns 6–9) + diversity penalty | `class_weight` |
| 3 | `CrossEntropyLoss` (3-class) | `weight` |

Class weighting is standard practice for imbalanced data (weights computed from
**training** label counts only — no test leakage) and is not "cheating."
