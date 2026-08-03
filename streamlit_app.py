"""
CSR interactive inference app for ISIC-2017 -- Streamlit.

Mirrors the layout of the authors' official demo (which runs on TBX11K chest
X-rays) but for our skin-lesion model:

  sidebar  : configuration -- checkpoint, patient source, interaction settings
  main     : the patient image + a drawable canvas for spatial boxes
             one row per concept: heatmap on the patient | matched atlas image
             a checkbox per concept to REJECT it (Sec. 3.3 concept interaction)
  bottom   : two bar charts, "Predict" vs "Interactive prediction"

All scoring goes through CSRNetwork.predict_interactive, so the numbers match
evaluate.py exactly -- this file only handles presentation and turning the
doctor's boxes into the importance map A of Eqns 12-13.

Run (locally):
    streamlit run streamlit_app.py

Run (Colab -- needs a tunnel):
    !pip install streamlit streamlit-drawable-canvas -q
    !npm install -g localtunnel
    !streamlit run streamlit_app.py --server.port 8501 &>/content/log.txt &
    !npx localtunnel --port 8501
    # open the printed URL; the tunnel password is the Colab VM's external IP:
    !curl -s https://loca.lt/mytunnelpassword
"""
import io
import os

import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from dataset import ISIC2017Dataset, CONCEPT_KEYS, CLASS_NAMES
from evaluate import build_csr_from_state
from models.concept_model import DEFAULT_BACKBONE
from utils import build_transforms, resolve_device, IMAGENET_MEAN, IMAGENET_STD

# streamlit-drawable-canvas calls streamlit.elements.image.image_to_url, which
# newer Streamlit moved to streamlit.elements.lib.image_utils. Re-export it so
# the canvas keeps working on both old and new Streamlit.
try:
    import streamlit.elements.image as _st_image
    if not hasattr(_st_image, "image_to_url"):
        try:
            from streamlit.elements.lib.image_utils import image_to_url as _i2u
        except ImportError:
            from streamlit.elements.lib.image_utils import _image_to_url as _i2u
        _st_image.image_to_url = _i2u
except Exception:
    pass

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except Exception:
    HAS_CANVAS = False

CONCEPT_LABELS = {
    "milia_like_cyst":  "Milia-like cysts",
    "pigment_network":  "Pigment network",
    "negative_network": "Negative network",
    "streaks":          "Streaks",
}
POS_RGBA = "rgba(0, 200, 0, 0.35)"      # positive box  -> A = 1
NEG_RGBA = "rgba(220, 0, 0, 0.35)"      # negative box  -> A = 0

st.set_page_config(page_title="CSR — Interactive Inference (ISIC)", layout="wide")


# --------------------------------------------------------------- model loading
@st.cache_resource(show_spinner="Loading CSR model…")
def load_everything(ckpt_dir, backbone, device_str):
    dev = resolve_device(device_str)
    state = torch.load(os.path.join(ckpt_dir, "csr_network_final_best.pth"),
                       map_location=dev)
    net, K, M = build_csr_from_state(state, backbone, dev)
    net.eval()

    atlas = torch.load(os.path.join(ckpt_dir, "atlas.pt"), map_location="cpu")
    tf = build_transforms(image_size=224, is_train=False)
    test_ds = ISIC2017Dataset("data/ISIC-2017_Test_v2_Data.zip",
                              "data/ISIC-2017_Test_v2_Part3_GroundTruth.csv",
                              "data/ISIC-2017_Test_v2_Part2_GroundTruth.zip", tf)
    train_ds = ISIC2017Dataset("data/isic_224x224.zip",
                               "data/ISIC-2017_Training_Part3_GroundTruth.csv",
                               "data/ISIC-2017_Training_Part2_GroundTruth.zip", tf)
    with torch.no_grad():
        _, _, S = net(torch.zeros(1, 3, 224, 224, device=dev))
    return net, atlas, test_ds, train_ds, dev, K, M, S.shape[-1]


def atlas_exemplar(atlas, train_ds, k, m):
    idx = int(atlas["best_image_idx"][k, m])
    ids = list(atlas["image_ids"])
    if idx < 0 or idx >= len(ids):
        return None, None
    img_id = ids[idx]
    try:
        return train_ds._open_image(img_id).resize((224, 224)), img_id
    except Exception:
        return None, img_id


# ------------------------------------------------------------------- rendering
def denorm(t):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    a = (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return (a * 255).astype(np.uint8)


def jet(x):
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], -1)


def overlay(base_u8, sim_map, alpha=0.45):
    m = sim_map.detach().float().cpu()[None, None]
    up = F.interpolate(m, size=base_u8.shape[:2], mode="bilinear",
                       align_corners=False)[0, 0].numpy()
    lo, hi = float(up.min()), float(up.max())
    up = (up - lo) / (hi - lo) if hi > lo else np.zeros_like(up)
    out = (1 - alpha) * (base_u8 / 255.0) + alpha * jet(up)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def boxes_to_importance(objects, grid, canvas_px, alpha):
    """
    Eqn 12: turn drawn rectangles into the (grid x grid) importance map A.
    Green boxes -> 1 (attend here), red -> 0 (ignore), unmarked -> alpha.
    A cell counts as covered if the box overlaps it at all.
    """
    if not objects:
        return None
    A = np.full((grid, grid), alpha, dtype=np.float32)
    cell = canvas_px / grid
    touched = False
    for o in objects:
        if o.get("type") != "rect":
            continue
        val = 1.0 if str(o.get("fill", "")).startswith("rgba(0, 200") else 0.0
        x0, y0 = o["left"], o["top"]
        x1 = x0 + o["width"] * o.get("scaleX", 1)
        y1 = y0 + o["height"] * o.get("scaleY", 1)
        c0, c1 = int(np.floor(x0 / cell)), int(np.ceil(x1 / cell))
        r0, r1 = int(np.floor(y0 / cell)), int(np.ceil(y1 / cell))
        A[max(r0, 0):min(r1, grid), max(c0, 0):min(c1, grid)] = val
        touched = True
    return torch.from_numpy(A) if touched else None


def prob_bars(p_before, p_after):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, p, title in zip(axes, [p_before, p_after],
                            ["Probability of disease (Predict)",
                             "Probability of disease (Interactive prediction)"]):
        ax.bar(CLASS_NAMES, p, color="#8b1a1a")
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=15)
    axes[0].set_ylabel("Probability Score")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- sidebar
st.sidebar.title("Configuration")
ckpt = st.sidebar.selectbox("Checkpoint", [
    "checkpoints_revive", "checkpoints_revive_div",
    "checkpoints_p1weighted_clsw", "checkpoints_deduped"], index=0)
label_mode = st.sidebar.selectbox("Label (for boxes you draw)", ["positive", "negative"])
alpha = st.sidebar.slider("Neutral weight α (paper: 0.2)", 0.0, 1.0, 0.2, 0.05)
device_str = st.sidebar.selectbox("Device", ["cuda", "cpu"], index=0)

net, atlas, test_ds, train_ds, dev, K, M, GRID = load_everything(
    ckpt, DEFAULT_BACKBONE, device_str)

st.sidebar.markdown("---")
src = st.sidebar.radio("Patient source", ["Test set", "Upload image"])
if src == "Test set":
    idx = st.sidebar.number_input("Patient index", 0, len(test_ds) - 1, 0, 1)
    img_t, concept_gt, target = test_ds[idx]
    gt_txt = CLASS_NAMES[int(target)]
else:
    up = st.sidebar.file_uploader("Dermoscopy image", type=["png", "jpg", "jpeg"])
    if up is None:
        st.info("Upload a dermoscopy image, or switch to the test set.")
        st.stop()
    pil = Image.open(up).convert("RGB")
    img_t = build_transforms(224, False)(pil)
    concept_gt, gt_txt = None, "n/a (uploaded)"

base_u8 = denorm(img_t)
x = img_t.unsqueeze(0).to(dev)

# ------------------------------------------------------------------- main
st.title("Concept-based Similarity Reasoning — Interactive Inference")
st.caption(f"ISIC-2017 · {K} concepts × {M} prototypes · feature grid {GRID}×{GRID} "
           f"· ground truth: **{gt_txt}**")

left, right = st.columns([1, 1])
def grid_fallback(base_u8, grid, alpha):
    """
    Cell-grid spatial input, used when the drawable canvas is unavailable.
    Each cell IS one feature-map location (32x32 px of the 224px image), so
    this is exactly as expressive as drawing boxes -- boxes get quantised to
    these cells anyway (Eqn 12).
    """
    if "cells" not in st.session_state or st.session_state.get("cells_grid") != grid:
        st.session_state.cells = np.zeros((grid, grid), dtype=int)
        st.session_state.cells_grid = grid
    st.image(base_u8, width=392)
    st.caption(f"Click cells to mark them **{label_mode}**  ·  "
               "✅ attend (A=1) · ⛔ ignore (A=0) · blank = neutral")
    val = 1 if label_mode == "positive" else -1
    for r in range(grid):
        cols = st.columns(grid)
        for c in range(grid):
            cur = int(st.session_state.cells[r, c])
            face = "✅" if cur == 1 else ("⛔" if cur == -1 else "·")
            if cols[c].button(face, key=f"cell_{r}_{c}"):
                st.session_state.cells[r, c] = 0 if cur == val else val
                st.rerun()
    if st.button("Clear cells"):
        st.session_state.cells[:] = 0
        st.rerun()
    cells = st.session_state.cells
    if not cells.any():
        return None
    A = np.full((grid, grid), alpha, dtype=np.float32)
    A[cells == 1] = 1.0
    A[cells == -1] = 0.0
    return torch.from_numpy(A)


with left:
    st.subheader("Spatial guidance")
    importance = None
    canvas_ok = False
    if HAS_CANVAS:
        try:
            st.caption(f"Draw **{label_mode}** boxes on the lesion. "
                       "Green = attend here (A=1), red = ignore (A=0).")
            canvas = st_canvas(
                fill_color=POS_RGBA if label_mode == "positive" else NEG_RGBA,
                stroke_width=2,
                stroke_color="#00c800" if label_mode == "positive" else "#dc0000",
                background_image=Image.fromarray(base_u8),
                drawing_mode="rect", key=f"canvas_{ckpt}_{src}",
                width=448, height=448, update_streamlit=True)
            objects = (canvas.json_data or {}).get("objects", []) if canvas else []
            importance = boxes_to_importance(objects, GRID, 448, alpha)
            canvas_ok = True
        except Exception as e:
            st.info(f"Drawable canvas unavailable ({type(e).__name__}); "
                    "using the cell grid instead.")
    if not canvas_ok:
        importance = grid_fallback(base_u8, GRID, alpha)

with right:
    st.subheader("Concept-level interaction")
    st.caption("Tick a concept to **reject** it — this zeroes all "
               f"{M} of its prototype scores for this patient.")
    rejected = []
    for k in range(K):
        key = CONCEPT_KEYS[k]
        present = ("—" if concept_gt is None else
                   ("present" if concept_gt[k] > 0.5 else "absent"))
        if st.checkbox(f"Reject **{CONCEPT_LABELS[key]}**  ·  ground truth: {present}",
                       key=f"rej_{k}"):
            rejected.append(k)

# ---- inference: baseline and interactive -------------------------------
with torch.no_grad():
    logits_b, _, S_maps = net(x)
    p_before = torch.softmax(logits_b.float(), 1)[0].cpu().numpy()

    logits_i, s_scores, S_maps_i = net.predict_interactive(
        x, importance_map=importance,
        rejected_concepts=rejected or None)
    p_after = torch.softmax(logits_i.float(), 1)[0].cpu().numpy()

st.markdown("---")
st.subheader("Evidence per concept")
cols = st.columns(K)
for k in range(K):
    with cols[k]:
        key = CONCEPT_KEYS[k]
        st.markdown(f"**{CONCEPT_LABELS[key]}**")
        if k in rejected:
            st.error("rejected by clinician")
            continue
        m_win = int(s_scores[0, k].argmax())
        st.caption(f"prototype #{m_win} · similarity {float(s_scores[0,k,m_win]):.3f}")
        st.image(overlay(base_u8, S_maps_i[0, k, m_win]),
                 caption="patient (evidence)", width=230)
        ex, ex_id = atlas_exemplar(atlas, train_ds, k, m_win)
        if ex is not None:
            st.image(ex, caption=f"atlas: {ex_id}", width=230)
        else:
            st.caption("_no atlas exemplar (unused prototype)_")

st.markdown("---")
c1, c2 = st.columns([2, 1])
with c1:
    st.pyplot(prob_bars(p_before, p_after))
with c2:
    b, a = int(p_before.argmax()), int(p_after.argmax())
    st.metric("Prediction", CLASS_NAMES[a],
              delta=None if a == b else f"was {CLASS_NAMES[b]}")
    st.write(f"confidence **{p_after[a]*100:.1f}%** "
             f"(before {p_before[b]*100:.1f}%)")
    margin = np.sort(p_after)[::-1]
    st.write(f"top1−top2 margin **{margin[0]-margin[1]:.3f}**"
             + ("  ← indecisive, review advised" if margin[0]-margin[1] < 0.3 else ""))
    if gt_txt not in ("n/a (uploaded)",):
        ok_b, ok_a = CLASS_NAMES[b] == gt_txt, CLASS_NAMES[a] == gt_txt
        if not ok_b and ok_a:
            st.success("Your interaction CORRECTED this prediction")
        elif ok_b and not ok_a:
            st.error("Your interaction BROKE a correct prediction")
