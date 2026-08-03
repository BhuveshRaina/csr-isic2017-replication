# CSR — Concept-based Similarity Reasoning (ISIC-2017 replication)

Reimplementation of **"Interactive Medical Image Analysis with Concept-based
Similarity Reasoning"** (Huy et al., CVPR 2025) on the ISIC-2017 skin-lesion
dataset, following the main paper and the supplementary material.

The pipeline has three training stages plus evaluation, interpretability and
doctor-in-the-loop interaction:

```
Phase 1  Concept model      F + concept head C, multi-label BCE over 4 concepts
Phase 2  Prototype learning  projector P + 400 prototypes, contrastive loss
Phase 3  Task head           single linear layer H, cross-entropy over 3 classes
Eval     macro-F1 on the official ISIC-2017 test set (Table 1)
Extras   atlas (Eqn 11), similarity-map explanations, spatial/concept interaction,
         Pointing Game
```

---

## 1. Why your numbers didn't match the paper

The single biggest issue was **evaluation methodology, not the model**:

| # | Problem in the original code | Effect | Fix |
|---|------------------------------|--------|-----|
| 1 | **No held-out test set.** Phase 3 computed macro-F1 on the *same 2000 training images* it was fitting. | You were reading a *training* score and comparing it to the paper's *test* score (71.5). These are different quantities. | `splits.py` builds proper train/val/test; `evaluate.py` reports macro-F1 on the official **ISIC-2017 Test v2 (600 imgs)**. |
| 2 | **Model selection on training loss.** Best checkpoint chosen by lowest train loss. | Saved an over-fit / mis-calibrated head. | All phases now select the best checkpoint on a **validation** metric (concept-F1 / proto-accuracy / macro-F1). |
| 3 | **Backbone.** Used `convnext_tiny.in12k_ft_in1k`. | Not what the supplementary specifies ("ConvNeXt-T, ImageNet pretrained" = IN-1k). Stronger, but a deviation. | Default is now the faithful `convnext_tiny.fb_in1k`; the IN-12k init stays available via `--backbone`. |
| 4 | **Missing ISIC-2019 pretraining.** Supplementary pretrains the ISIC baseline on ISIC-2019 (~25–35k imgs) before ISIC-2017. | A few F1 points of headroom you can't get from ImageNet alone. | Documented; ImageNet init used by default. See §5. |
| 5 | **No evaluation / interpretability / interaction code at all.** Project only trained. | Couldn't reproduce Table 1, Table 2, or the interaction results. | Added `evaluate.py`, `build_atlas.py`, `explain.py`, `interaction_demo.py`, `pointing_game.py`. |

The core **model math was mostly correct** (concept head, Eqn 2 local vectors,
Eqns 6–9 contrastive loss, Eqn 10 max-similarity, Eqn 1 task head). Those were
kept and cleaned up, not rewritten. Verified item-by-item in §4.

---

## 2. Setup

```bash
pip install -r requirements.txt
```

You already have the **training** split in `data/`:
`isic_224x224.zip`, `ISIC-2017_Training_Part2_GroundTruth.zip`,
`ISIC-2017_Training_Part3_GroundTruth.csv`.

To reproduce the paper's number you also need the official **test** split
(the paper reports on ISIC-2017 Test v2, 600 images). Download it:

```bash
bash download_isic2017.sh data
```

This fetches (from `https://isic-archive.s3.amazonaws.com/challenges/2017/`):

- `ISIC-2017_Validation_Data.zip` / `_Part2_GroundTruth.zip` / `_Part3_GroundTruth.csv`
- `ISIC-2017_Test_v2_Data.zip` / `_Part2_GroundTruth.zip` / `_Part3_GroundTruth.csv`

> The full-res Data zips also contain `*_superpixels.png`, which the Pointing
> Game needs. The 224px training zip does not.

---

## 3. Run the full pipeline

Common data flags (define once, reuse):

```bash
TRAIN="--img_dir data/isic_224x224.zip \
  --task2_dir data/ISIC-2017_Training_Part2_GroundTruth.zip \
  --task3_csv data/ISIC-2017_Training_Part3_GroundTruth.csv"

VAL="--val_img data/ISIC-2017_Validation_Data.zip \
  --val_task2 data/ISIC-2017_Validation_Part2_GroundTruth.zip \
  --val_task3 data/ISIC-2017_Validation_Part3_GroundTruth.csv"

TEST="--test_img data/ISIC-2017_Test_v2_Data.zip \
  --test_task2 data/ISIC-2017_Test_v2_Part2_GroundTruth.zip \
  --test_task3 data/ISIC-2017_Test_v2_Part3_GroundTruth.csv"
```

```bash
# Phase 1 — concept model
python train_phase1.py $TRAIN $VAL --epochs 50 --batch_size 32 --lr 1e-4

# Phase 2 — projector + prototypes
python train_phase2.py $TRAIN $VAL --epochs 30 --lr 1e-4 \
  --phase1_weights checkpoints/concept_model_phase1_best.pth

# Phase 3 — task head
python train_phase3.py $TRAIN $VAL --epochs 50 --lr 1e-3 \
  --phase1_weights checkpoints/concept_model_phase1_best.pth \
  --phase2_weights checkpoints/phase2_best.pth

# Evaluate on the official test set (Table 1)
python evaluate.py $TRAIN $TEST \
  --weights checkpoints/csr_network_final_best.pth
```

If you **omit** all `--val_*`/`--test_*` flags, the code automatically falls
back to a stratified 70/15/15 split of the training set (a legitimate held-out
metric, but on a different test set than the paper — so it won't equal 71.5).

### Interpretability & interaction (full-paper extras)

```bash
# Prototype projection / concept atlas (Eqn 11)
python build_atlas.py $TRAIN \
  --phase1_weights checkpoints/concept_model_phase1_best.pth \
  --phase2_weights checkpoints/phase2_best.pth

# Similarity-map explanation for one image (Sec 3.1)
python explain.py --weights checkpoints/csr_network_final_best.pth \
  --image path/to/image.jpg --atlas checkpoints/atlas.pt --out explanation.png

# Doctor-in-the-loop interaction (Sec 3.3, Eqn 12-13)
python interaction_demo.py --weights checkpoints/csr_network_final_best.pth \
  --image path/to/image.jpg --pos_box 0.1 0.1 0.5 0.5 --reject_concept 2

# Pointing Game (needs the full-res test Data zip with superpixels)
python pointing_game.py --weights checkpoints/csr_network_final_best.pth \
  --test_img data/ISIC-2017_Test_v2_Data.zip \
  --test_task2 data/ISIC-2017_Test_v2_Part2_GroundTruth.zip
```

CPU/MPS work too — pass `--device cpu` (or `mps`); AMP auto-disables off CUDA.

---

## 4. Faithfulness checklist (paper → code)

| Paper | Where | Status |
|-------|-------|--------|
| F = ConvNeXt-T, ImageNet pretrained | `models/concept_model.py` | ✔ default `convnext_tiny.fb_in1k` |
| Concept head C = 1×1 conv (no bias) → GAP → Sigmoid | `models/concept_model.py` | ✔ |
| Phase 1 loss = multi-label BCE | `train_phase1.py` | ✔ |
| Local concept vector, Eqn 2 (spatial softmax · f, summed) | `train_phase2.py:extract_local_concept_vectors` | ✔ vectorised |
| Projector P = 3× (Conv1d + IBN + ReLU) residual, L2-norm out | `models/projector_model.py` | ✔ |
| M = 100 prototypes per concept | `ConceptPrototypes` | ✔ (K·M = 400) |
| Contrastive loss, Eqns 6–9 (λ=20, γ=1000, δ=0.1) | `MultiPrototypeContrastiveLoss` | ✔ |
| Inference: s_km = max_{h,w} ⟨p_km, P(f(h,w))⟩, Eqn 10 | `models/csr_network.py:similarity_maps` | ✔ |
| Task head H = single Linear, cross-entropy, Eqn 1 | `train_phase3.py` | ✔ |
| Metric = macro-F1 (class imbalance) | `evaluate.py` | ✔ |
| #prototypes 400, #exp 4 (Table 1 ISIC row) | `evaluate.py` prints both | ✔ |
| Prototype projection, Eqn 11 | `build_atlas.py` | ✔ (uses argmax; the paper prints argmin, a sign typo) |
| Spatial interaction, Eqns 12–13; concept rejection | `csr_network.predict_interactive` | ✔ |
| Pointing Game | `pointing_game.py` | ✔ (ISIC concept masks via superpixels) |

### Known, deliberate deviations
- **ISIC-2019 pretraining not included.** The supplementary pretrains the ISIC
  baseline on ISIC-2019 first; we start from ImageNet. Expect to land a little
  below 71.5. To close the gap, pretrain the Phase-1 concept model on ISIC-2019
  (download link in `download_isic2017.sh`) and load it into Phase 1.
- **IBN on 1-D concept vectors.** Concept vectors have no spatial extent, so the
  InstanceNorm branch degenerates to per-sample (vector) normalisation and the
  BatchNorm branch relies on running stats computed on concept vectors while
  inference feeds all patches — a mild train/test distribution shift inherent to
  the paper's projector design. Kept faithful; noted here.
- **3 classes = {melanoma, seborrheic keratosis, nevus}** — the standard
  ISIC-2017 Task-3 labelling (the supplementary phrases this loosely).

---

## 5. Files

```
utils.py              seeding, device, AMP helpers, transforms
dataset.py            ISIC2017Dataset (splits/subset/official test) + stratified_split
splits.py             resolve train/val/test sources (official or internal)
models/concept_model.py     F + concept head C
models/projector_model.py   Projector P, ConceptPrototypes, contrastive loss
models/csr_network.py       full CSR + interaction (Eqn 10-13)
train_phase1.py / _phase2.py / _phase3.py    staged training w/ validation
evaluate.py           test-set macro-F1, per-class F1, confusion matrix (Table 1)
build_atlas.py        prototype projection / concept atlas (Eqn 11)
explain.py            similarity-map explanations (Sec 3.1)
interaction_demo.py   spatial + concept-level interaction (Sec 3.3)
pointing_game.py      Pointing Game hit-rate (Table 2 analogue)
download_isic2017.sh  fetch official validation + test archives
```

---

## 6. Results & doctor-in-the-loop findings

- **Best static model:** macro-F1 **65.66** on the official ISIC-2017 test set
  (`checkpoints_revive`), vs the paper's 71.5 — the main documented gap is the
  missing ISIC-2019 pretraining plus the Phase-1 concept ceiling (~0.68).
- **Doctor-in-the-loop:** the paper only tests interaction on TBX11K
  (chest X-ray), never on ISIC. Testing concept-level rejection on ISIC here
  showed the paper's zeroing operation is harmful; an in-distribution
  replacement makes it safe; and **concept-dropout retraining** produces the
  first net-positive interaction on ISIC (**64.28 → 64.94**, +0.66 at the
  paper's own margin < 0.3 trigger).

Full write-ups:

- **`EVALUATION_METRICS.md`** — every metric used, defined, with code locations.
- **`PROJECT_NOTES.md`** — key facts, the paper's 71.5 in context, problems
  solved (prototype collapse, class imbalance), and the full doctor-in-the-loop
  investigation with honest caveats and open next steps.

### Interaction / analysis scripts

```
test_time_interaction.py     oracle concept-rejection (Sec 3.3), whole test set
confidence_gate.py           paper's margin<0.3 trigger swept
confidence_delta_experiment  per-image confidence delta, stratified by margin
prototype_impact.py          per-prototype ablation "safety map"
improve_concept_rejection.py in-distribution replacement strategies compared
selective_rejection.py       selective + in-distribution rejection sweep
find_wrong_predictions.py    list wrong cases for interactive testing
doctor_ui.py / streamlit_app.py   interactive doctor-in-the-loop interfaces
```
