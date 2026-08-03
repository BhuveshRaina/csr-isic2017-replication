# Project Notes — Key Facts & Findings

A running record of the important facts, decisions, and results for this
replication of **"Interactive Medical Image Analysis with Concept-based
Similarity Reasoning" (CSR, Huy et al., CVPR 2025)** on ISIC-2017.

For metric definitions see `EVALUATION_METRICS.md`. For setup/usage see
`README.md`.

---

## 1. What the model is

CSR classifies a skin lesion by comparing it to a set of **concept prototypes**
and reasoning over the similarities, so every prediction comes with human-
readable evidence ("this looks like these known examples of pigment network").

- **4 concepts** × **100 prototypes** = **400 prototypes** total.
- **3 disease classes**: seborrheic keratosis, melanoma, nevus.
- **3 training phases:**
  - Phase 1 — Concept model (backbone + concept head), multi-label BCE.
  - Phase 2 — Projector + 400 prototypes, multi-prototype contrastive loss.
  - Phase 3 — Single linear task head, cross-entropy over 3 classes.
- Everything before the task head is frozen when training later phases.

---

## 2. The paper's headline number, in context

- The paper reports **CSR = 71.5 macro-F1** on ISIC-2017 (400 prototypes,
  explanation size 4).
- **This is benchmarked only against other interpretable / concept-based
  methods** (CBM 45.5, ProtoPNet 38.5, ProtoPool 31.0, ProtoTree 66.1, PIP-Net
  69.9). **There is no black-box baseline in the paper's Table 1.** So 71.5 is
  "best among interpretable models," not "state of the art over all AI."
- CSR's real advantage is **explanation size 4** (vs 90–600 for the others) at
  competitive-or-best accuracy.
- The 71.5 is on ISIC-2017 (confirmed in the supplementary, Sec 3.3). The paper
  additionally pretrains its *baselines* on the larger ISIC-2019 set (35k
  images); that is a baseline detail, not the eval set.

---

## 3. Our results

| Model | Macro-F1 (test) | Notes |
|-------|-----------------|-------|
| Best static model (`checkpoints_revive`) | **65.66** | prototype revival + class-weighted losses |
| Concept-dropout model (`checkpoints_dropout`) | 64.28 clean | trained to support intervention |
| Concept-dropout **+ correct doctor** | **64.94** | +0.66 from interaction, paper's own trigger |

~5.8 points below the paper's 71.5, with the main documented gap being the
**missing ISIC-2019 pretraining** (see README §4 "Known deviations") plus the
Phase-1 concept-detection ceiling (~0.68 concept-F1).

---

## 4. Why our first numbers were far below 71.5 (methodology, not model)

The biggest early problem was evaluation, not the network (full table in
`README.md` §1):

1. **No held-out test set** — Phase 3 was scoring macro-F1 on the same 2000
   training images it was fitting, then comparing to the paper's *test* number.
2. **Model selection on training loss** — saved over-fit / mis-calibrated heads.
   Fixed: every phase now selects its checkpoint on a **validation metric**.
3. **Backbone mismatch** and **no evaluation/interpretability code** at all.

The core model math (concept head, local vectors, contrastive loss, max-
similarity, task head) was mostly correct and was kept.

---

## 5. Problems solved along the way

- **Validation loss rising while F1 improves** — expected and harmless.
  Log-loss punishes confident-but-wrong predictions far more than unconfident-
  wrong ones, so calibration can worsen while ranking (F1) improves. We always
  select on F1, never on raw loss, so this never affected the saved model.
- **Prototype collapse** — the assignment softmax (γ=1000) is effectively an
  argmax, so only the single winning prototype per concept got gradient
  ("rich-get-richer"). Only 23–42 of 400 prototypes were alive. Fixed with
  **random-restart revival** (VQ-VAE style): periodically reseed dead prototypes
  onto poorly-covered training vectors. Alive count → 289–316 of 400, and test
  macro-F1 rose to the best 65.66.
- **Class imbalance** — handled with class-weighted losses in all 3 phases.
  Legitimate (weights from training counts only), not cheating.

---

## 6. Doctor-in-the-loop — the main investigation

The paper claims two interaction mechanisms let a doctor improve predictions at
test time:
- **Concept-level**: reject a concept → its similarity scores are zeroed.
- **Spatial-level** (the paper's *novel* contribution): draw boxes to reweight
  where the model looks, before max-pooling.

**Crucially, the paper only ever tests interaction on TBX11K (chest X-ray),
never on ISIC.** Its Table 3 shows a maximum gain of **+0.7 macro-F1** (M.D.
student, concept-level, baseline 94.4 → 95.1), with a single M.D. student and a
single non-expert, and only when the model is indecisive (margin < 0.3, α = 0.2).
So there is **no published ISIC interaction result** — we are the first to test
it there.

### What we found (concept-level rejection on ISIC)

1. **Zeroing a concept is out-of-distribution and harmful.** The task head only
   ever saw similarity scores in a narrow band (~0.09 mean) during training and
   never an exact zero. Forcing 100 scores to zero — especially a high, over-
   firing one — produces violent, mostly-wrong swings. Blanket zeroing: **net
   −650** across the test set.

2. **Replacing instead of deleting makes it safe.** Substituting a rejected
   concept with its *typical-absent* profile (learned from training) keeps the
   classifier in-distribution. Blanket `absent_median`: **net −8** (≈ neutral),
   and the average effect on the correct class flips from −12pp to +2pp.

3. **The paper's own trigger still fails on the static model.** Even gating
   exactly as the paper specifies (margin < 0.3), concept rejection on the
   65.66 model is net-negative (at T=0.3: 21 fixed, 50 broke, F1 → 60.04).

4. **Atlas view ≠ classifier reliance (novel finding).** How often a prototype
   wins the atlas competition (`hit_count`) is a *different, largely uncorrelated*
   statistic from how much the classifier depends on it. A prototype can be
   atlas-rare yet critical to the prediction. This is a real gap in the paper's
   premise that doctors can safely review prototypes through the atlas.

5. **Selective, in-distribution rejection is safe but not enough on the static
   model.** Rejecting only over-firing concepts, only when indecisive, using
   `absent_median` — no policy beat baseline. This proved the ceiling is in the
   **model**, not the policy: a fixed linear head drops a term, it never
   re-reasons or shifts attention to the remaining concepts.

6. **Concept-dropout retraining makes a correct doctor finally help.** Training
   the task head with truly-absent concepts randomly removed teaches it to lean
   on the remaining concepts when one is rejected. On this model, the paper's own
   margin < 0.3 trigger gives the **first net-positive interaction on ISIC:
   64.28 → 64.94 (+0.66), 21 fixed vs 17 broke.** The winning policy is the
   least-tuned one (paper's threshold, simplest gate), which makes it credible.

### The realistic ceiling
An oracle that intervened only on already-wrong cases fixed **~40% of errors
(+11 F1)** — the theoretical headroom for concept-level correction. The gap
between +0.66 (realistic) and +11 (oracle) is the room a better trigger could
still recover.

### Honest caveats
- The +0.66 model (64.94) is still **below** the best static model (65.66):
  "a model built for interaction, once corrected, approaches but doesn't yet beat
  the best static model."
- The +0.66 was the best cell of a threshold sweep evaluated on the test set. To
  be fully rigorous, the threshold should be picked on **validation** and applied
  once to test. (Open next step.)
- The **spatial** interaction mechanism (the paper's headline) is implemented
  (`predict_interactive`) but **not yet quantitatively evaluated** on ISIC. A
  legitimate spatial oracle exists: ISIC-2017 Part-2 superpixel masks. (Open.)

---

## 7. Open next steps

1. Validation-selected single-policy confirmation of the +0.66 (remove test-set
   tuning concern).
2. Try gentler concept-dropout (p = 0.3) to keep more of the 65.66 baseline.
3. Quantitatively evaluate **spatial** interaction using Part-2 superpixel masks
   as the oracle — the paper's untested headline mechanism.
4. (Longer term) React frontend calling the Python model as a backend;
   `predict_interactive` is the clean API surface.

---

## 8. Deliverables / interfaces built

- `evaluate.py` — Table 1 reproduction (macro-F1, per-class, confusion matrix).
- `build_atlas.py`, `explain.py` — concept atlas + similarity-map explanations.
- `pointing_game.py` — trustworthiness (Table 2 analogue).
- `doctor_ui.py` — Colab ipywidgets doctor-in-the-loop prototype.
- `streamlit_app.py` — full interactive web app mirroring the official CSR demo.
- Interaction experiments: `test_time_interaction.py`, `confidence_gate.py`,
  `confidence_delta_experiment.py`, `prototype_impact.py`,
  `improve_concept_rejection.py`, `selective_rejection.py`,
  `find_wrong_predictions.py`.
