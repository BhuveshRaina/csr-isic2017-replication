"""
Doctor-in-the-loop interface for CSR (paper Sec. 3.2 / 3.3).

Runs in a Jupyter/Colab notebook via ipywidgets. Implements all three
interaction modes described in the paper:

  1. Train-time atlas refinement (Sec. 3.2)
     The doctor reviews the prototype atlas -- for each prototype p_km, the
     nearest training image I(p_km) (Eqn 11) -- and discards prototypes whose
     exemplar image shows a shortcut / artefact rather than real pathology.
     A discarded prototype has s_km forced to 0 for *every* future prediction.

  2. Test-time concept rejection (Sec. 3.3)
     For one patient, the doctor rejects a concept k they judge absent. All M
     scores s_km for that concept are zeroed for that single prediction.

  3. Test-time spatial interaction (Eqns 12-13)
     The doctor marks regions of the 7x7 feature grid as positive (A=1),
     negative (A=0) or neutral (A=alpha). Every one of the K*M similarity maps
     is multiplied element-wise by A *before* max-pooling, so the winning
     location -- and therefore the prediction -- can change.

All inference is delegated to CSRNetwork.predict_interactive / .forward so that
this file never re-implements the scoring maths. That guarantees the numbers
shown here match evaluate.py exactly.

Usage in a notebook:

    from doctor_ui import launch_ui
    ui = launch_ui(
        weights="checkpoints_deduped/csr_network_final_best.pth",
        atlas="checkpoints_deduped/atlas.pt",
        train_img="data/isic_224x224.zip",
        train_task2="data/ISIC-2017_Training_Part2_GroundTruth.zip",
        train_task3="data/ISIC-2017_Training_Part3_GroundTruth.csv",
        test_img="data/ISIC-2017_Test_v2_Data.zip",
        test_task2="data/ISIC-2017_Test_v2_Part2_GroundTruth.zip",
        test_task3="data/ISIC-2017_Test_v2_Part3_GroundTruth.csv",
    )
"""
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import ipywidgets as widgets
from IPython.display import display

from dataset import ISIC2017Dataset, CONCEPT_KEYS, CLASS_NAMES
from evaluate import build_csr_from_state          # reuse: no duplicated model wiring
from models.concept_model import DEFAULT_BACKBONE
from utils import build_transforms, resolve_device, IMAGENET_MEAN, IMAGENET_STD

# Human-readable concept names for the clinician-facing labels.
CONCEPT_LABELS = {
    "milia_like_cyst":  "Milia-like cysts",
    "pigment_network":  "Pigment network",
    "negative_network": "Negative network",
    "streaks":          "Streaks",
}


# --------------------------------------------------------------------- loading
class Pipeline:
    """Everything the UI needs: model, atlas, and the two image sources."""

    def __init__(self, net, atlas, test_ds, train_ds, device, grid):
        self.net, self.atlas = net, atlas
        self.test_ds, self.train_ds = test_ds, train_ds
        self.device, self.grid = device, grid
        self.K, self.M = int(atlas["K"]), int(atlas["M"])
        # map atlas image index -> position in the train dataset we can index
        self.atlas_ids = list(atlas["image_ids"])
        self._train_pos = {name: i for i, name in enumerate(train_ds.img_names)}

    def exemplar_image(self, k, m):
        """The training image linked to prototype (k, m) by Eqn 11, or None."""
        idx = int(self.atlas["best_image_idx"][k, m])
        if idx < 0 or idx >= len(self.atlas_ids):
            return None, None
        img_id = self.atlas_ids[idx]
        pos = self._train_pos.get(img_id)
        if pos is None:
            return None, img_id
        return self.train_ds._open_image(img_id).resize((224, 224)), img_id


def load_pipeline(weights, atlas, train_img, train_task2, train_task3,
                  test_img, test_task2, test_task3,
                  backbone=DEFAULT_BACKBONE, device="cuda", image_size=224):
    dev = resolve_device(device)

    state = torch.load(weights, map_location=dev)
    net, K, M = build_csr_from_state(state, backbone, dev)
    net.eval()

    atlas_obj = torch.load(atlas, map_location="cpu")
    if int(atlas_obj["K"]) != K or int(atlas_obj["M"]) != M:
        raise ValueError(
            f"Atlas is (K={atlas_obj['K']}, M={atlas_obj['M']}) but the network is "
            f"(K={K}, M={M}). The atlas must be built from the same Phase-2 "
            f"checkpoint that produced these weights.")

    eval_tf = build_transforms(image_size=image_size, is_train=False)
    test_ds = ISIC2017Dataset(test_img, test_task3, test_task2, eval_tf)
    train_ds = ISIC2017Dataset(train_img, train_task3, train_task2, eval_tf)

    # feature-grid size (7 for 224px ConvNeXt-T); derive it rather than assume.
    with torch.no_grad():
        probe = torch.zeros(1, 3, image_size, image_size, device=dev)
        _, _, S = net(probe)
        grid = S.shape[-1]

    print(f"Loaded CSR: K={K} concepts x M={M} prototypes | feature grid {grid}x{grid}")
    print(f"Test images: {len(test_ds)} | Train (atlas) images: {len(train_ds)}")
    return Pipeline(net, atlas_obj, test_ds, train_ds, dev, grid)


# --------------------------------------------------------------------- imaging
def denormalize(tensor):
    """(3,H,W) normalized tensor -> uint8 HWC array."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    arr = (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)


def _jet(x):
    """Minimal jet colormap on x in [0,1] -> (...,3) float, avoids a mpl import."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay_heatmap(base_uint8, sim_map, alpha=0.45):
    """
    Upsample a (grid,grid) similarity map to the image size, colorize, blend.
    Normalized per-map so the winning region is always visible.
    """
    m = sim_map.detach().float().cpu()[None, None]
    up = F.interpolate(m, size=base_uint8.shape[:2], mode="bilinear",
                       align_corners=False)[0, 0].numpy()
    lo, hi = float(up.min()), float(up.max())
    up = (up - lo) / (hi - lo) if hi > lo else np.zeros_like(up)
    heat = _jet(up)
    blended = (1 - alpha) * (base_uint8 / 255.0) + alpha * heat
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def to_png_widget(arr_or_pil, width=200):
    img = arr_or_pil if isinstance(arr_or_pil, Image.Image) else Image.fromarray(arr_or_pil)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return widgets.Image(value=buf.getvalue(), format="png",
                         layout=widgets.Layout(width=f"{width}px", height=f"{width}px"))


# ------------------------------------------------------------------------- UI
class DoctorUI:
    """
    Interaction state
    -----------------
    discarded   : set of (k, m) removed during atlas review        -> Sec. 3.2
    rejected    : set of concept indices k rejected for this case  -> Sec. 3.3
    cell_state  : (grid, grid) int in {0:neutral, 1:positive, -1:negative} -> Eqn 12
    """

    CELL_CYCLE = {0: 1, 1: -1, -1: 0}
    CELL_STYLE = {0: ("", ""), 1: ("success", "+"), -1: ("danger", "-")}

    def __init__(self, pipe, alpha=0.3):
        self.p = pipe
        self.alpha = alpha
        self.discarded = set()
        self.rejected = set()
        self.cell_state = np.zeros((pipe.grid, pipe.grid), dtype=int)
        self._build()

    # ------------------------------------------------------------ state -> tensors
    def _prototype_mask(self):
        if not self.discarded:
            return None
        mask = torch.ones(self.p.K, self.p.M)
        for k, m in self.discarded:
            mask[k, m] = 0.0
        return mask

    def _importance_map(self):
        if not self.cell_state.any():
            return None
        A = np.full_like(self.cell_state, self.alpha, dtype=np.float32)
        A[self.cell_state == 1] = 1.0
        A[self.cell_state == -1] = 0.0
        return torch.from_numpy(A)

    # --------------------------------------------------------------- inference
    @torch.no_grad()
    def _infer(self):
        img, concepts, target = self.p.test_ds[self.idx_slider.value]
        x = img.unsqueeze(0).to(self.p.device)
        logits, s_scores, S_maps = self.p.net.predict_interactive(
            x,
            importance_map=self._importance_map(),
            rejected_concepts=sorted(self.rejected) or None,
            prototype_mask=self._prototype_mask(),
        )
        probs = torch.softmax(logits.float(), dim=1)[0].cpu()
        return img, concepts, int(target), probs, s_scores[0].float().cpu(), S_maps[0].float().cpu()

    # ------------------------------------------------------------------ render
    @torch.no_grad()
    def _baseline_probs(self):
        """Prediction with NO interaction, for side-by-side comparison."""
        img, _, _ = self.p.test_ds[self.idx_slider.value]
        x = img.unsqueeze(0).to(self.p.device)
        logits, _, _ = self.p.net(x)
        return torch.softmax(logits.float(), dim=1)[0].cpu()

    def _refresh(self, *_):
        img, gt_concepts, target, probs, s_scores, S_maps = self._infer()
        base = denormalize(img)
        base_probs = self._baseline_probs()

        pred = int(probs.argmax())
        base_pred = int(base_probs.argmax())
        active = bool(self.rejected or self.cell_state.any() or self.discarded)

        rows = [f"<b>Prediction:</b> {CLASS_NAMES[pred]} ({probs[pred]*100:.1f}%)",
                f"<b>Ground truth:</b> {CLASS_NAMES[target]}"]
        if active:
            flip = ("&nbsp;&nbsp;<b>CHANGED</b> "
                    f"({CLASS_NAMES[base_pred]} &rarr; {CLASS_NAMES[pred]})"
                    if pred != base_pred else "&nbsp;&nbsp;(unchanged)")
            correct_before = "correct" if base_pred == target else "wrong"
            correct_after = "correct" if pred == target else "wrong"
            rows.append(f"<b>Before interaction:</b> {CLASS_NAMES[base_pred]} "
                        f"({base_probs[base_pred]*100:.1f}%) — was {correct_before}, "
                        f"now {correct_after}{flip}")
        rows.append("<br><b>Class probabilities</b>"
                    + ("  (before &rarr; after)" if active else ""))
        for i, name in enumerate(CLASS_NAMES):
            bar = "&#9608;" * int(round(probs[i].item() * 20))
            if active:
                d = (probs[i] - base_probs[i]).item() * 100
                rows.append(f"{name:<22s} {base_probs[i]*100:5.1f}% &rarr; "
                            f"{probs[i]*100:5.1f}% ({d:+5.1f}) {bar}")
            else:
                rows.append(f"{name:<22s} {probs[i]*100:5.1f}% {bar}")
        note = []
        if self.rejected:
            note.append("concepts rejected: " + ", ".join(
                CONCEPT_LABELS[CONCEPT_KEYS[k]] for k in sorted(self.rejected)))
        if self.cell_state.any():
            note.append(f"spatial mask active ({int((self.cell_state==1).sum())} positive, "
                        f"{int((self.cell_state==-1).sum())} negative cells)")
        if self.discarded:
            note.append(f"{len(self.discarded)} prototypes discarded from atlas")
        if note:
            rows.append("<br><i>Active interactions: " + "; ".join(note) + "</i>")
        self.pred_box.value = "<pre style='font-family:monospace'>" + "\n".join(rows) + "</pre>"

        # one panel per concept: winning prototype's heatmap + its atlas exemplar
        panels = []
        for k in range(self.p.K):
            key = CONCEPT_KEYS[k]
            present = "present" if gt_concepts[k] > 0.5 else "absent"
            if k in self.rejected:
                panels.append(widgets.VBox([
                    widgets.HTML(f"<b>{CONCEPT_LABELS[key]}</b><br>"
                                 f"<span style='color:#b00'>REJECTED by clinician</span><br>"
                                 f"<small>ground truth: {present}</small>")],
                    layout=widgets.Layout(border="1px solid #ddd", padding="6px",
                                          margin="4px", width="440px")))
                continue

            m_win = int(s_scores[k].argmax())
            score = float(s_scores[k, m_win])
            heat = overlay_heatmap(base, S_maps[k, m_win])
            exemplar, ex_id = self.p.exemplar_image(k, m_win)

            imgs = [widgets.VBox([to_png_widget(heat),
                                  widgets.HTML("<small>patient (evidence)</small>")])]
            if exemplar is not None:
                imgs.append(widgets.VBox([to_png_widget(exemplar),
                                          widgets.HTML(f"<small>atlas: {ex_id}</small>")]))
            else:
                imgs.append(widgets.HTML("<small><i>no atlas exemplar<br>"
                                         "(dead prototype)</i></small>"))

            already = (k, m_win) in self.discarded
            discard_btn = widgets.Button(
                description=("discarded" if already else f"Discard  k={k}, m={m_win}"),
                button_style=("" if already else "warning"), disabled=already,
                layout=widgets.Layout(width="190px"))
            discard_btn.on_click(self._make_discard(k, m_win))

            panels.append(widgets.VBox([
                widgets.HTML(f"<b>{CONCEPT_LABELS[key]}</b> &nbsp; "
                             f"prototype #{m_win} &nbsp; similarity {score:.3f}<br>"
                             f"<small>ground truth: {present}</small>"),
                widgets.HBox(imgs), discard_btn],
                layout=widgets.Layout(border="1px solid #ddd", padding="6px",
                                      margin="4px", width="440px")))

        self.panel_box.children = tuple(panels)

    def _make_discard(self, k, m):
        def handler(_):
            self.discarded.add((k, m))
            self._refresh()
        return handler

    # ------------------------------------------------------------------- build
    def _build(self):
        p = self.p
        self.idx_slider = widgets.IntSlider(value=0, min=0, max=len(p.test_ds) - 1,
                                            description="Patient:", continuous_update=False,
                                            layout=widgets.Layout(width="520px"))
        self.idx_slider.observe(self._refresh, names="value")

        self.concept_boxes = []
        for k in range(p.K):
            cb = widgets.Checkbox(value=False, indent=False,
                                  description=CONCEPT_LABELS[CONCEPT_KEYS[k]],
                                  layout=widgets.Layout(width="190px"))
            cb.observe(self._make_reject(k), names="value")
            self.concept_boxes.append(cb)

        grid_buttons, rows = [], []
        for i in range(p.grid):
            row = []
            for j in range(p.grid):
                b = widgets.Button(description="", layout=widgets.Layout(
                    width="34px", height="34px", padding="0px", margin="1px"))
                b.on_click(self._make_cell(i, j, b))
                row.append(b)
                grid_buttons.append(b)
            rows.append(widgets.HBox(row))
        self._grid_buttons = grid_buttons

        clear_btn = widgets.Button(description="Clear boxes", button_style="info")
        clear_btn.on_click(self._clear_cells)
        reset_btn = widgets.Button(description="Reset all interactions", button_style="danger")
        reset_btn.on_click(self._reset_all)
        save_btn = widgets.Button(description="Save discarded prototypes",
                                  button_style="success",
                                  layout=widgets.Layout(width="220px"))
        save_btn.on_click(self._save_discards)
        self.save_msg = widgets.HTML()

        self.pred_box = widgets.HTML()
        self.panel_box = widgets.HBox([], layout=widgets.Layout(flex_flow="row wrap"))

        controls = widgets.VBox([
            widgets.HTML("<h3>Doctor-in-the-loop CSR</h3>"),
            self.idx_slider,
            widgets.HTML("<b>Reject a concept</b> (zeroes all "
                         f"{p.M} of its prototype scores for this case)"),
            widgets.HBox(self.concept_boxes),
            widgets.HTML(f"<b>Spatial guidance</b> — click cells of the {p.grid}x{p.grid} "
                         "feature grid to cycle neutral &rarr; "
                         "<span style='color:green'>positive (A=1)</span> &rarr; "
                         f"<span style='color:#b00'>negative (A=0)</span>. "
                         f"Unmarked cells get A={self.alpha} once any cell is marked."),
            widgets.VBox(rows),
            widgets.HBox([clear_btn, reset_btn, save_btn]),
            self.save_msg,
        ])

        self.widget = widgets.VBox([controls, widgets.HTML("<hr>"),
                                    self.pred_box, self.panel_box])
        self._refresh()

    def _make_reject(self, k):
        def handler(change):
            (self.rejected.add if change["new"] else self.rejected.discard)(k)
            self._refresh()
        return handler

    def _make_cell(self, i, j, button):
        def handler(_):
            self.cell_state[i, j] = self.CELL_CYCLE[int(self.cell_state[i, j])]
            style, label = self.CELL_STYLE[int(self.cell_state[i, j])]
            button.button_style, button.description = style, label
            self._refresh()
        return handler

    def _save_discards(self, _, path="checkpoints_deduped/discarded_prototypes.pt"):
        """Persist the curated mask so evaluate_curated.py can measure its effect."""
        mask = self._prototype_mask()
        if mask is None:
            mask = torch.ones(self.p.K, self.p.M)
        torch.save({"prototype_mask": mask,
                    "discarded": sorted(self.discarded)}, path)
        self.save_msg.value = (f"<span style='color:green'>Saved {len(self.discarded)} "
                               f"discarded prototypes &rarr; {path}</span>")

    def _clear_cells(self, _):
        self.cell_state[:] = 0
        for b in self._grid_buttons:
            b.button_style, b.description = "", ""
        self._refresh()

    def _reset_all(self, _):
        self.discarded.clear()
        self.rejected.clear()
        for cb in self.concept_boxes:
            cb.unobserve_all()
        for k, cb in enumerate(self.concept_boxes):
            cb.value = False
            cb.observe(self._make_reject(k), names="value")
        self._clear_cells(None)


# ------------------------------------------------------- atlas review (Sec 3.2)
class AtlasReview:
    """
    One-time train-time review of the whole atlas.

    Shows every *alive* prototype (hit_count > 0) with its exemplar image
    I(p_km) from Eqn 11, grouped by concept. Click a card to toggle
    keep/discard. Discarded prototypes get s_km = 0 permanently.

    This is the efficient way to do Sec. 3.2 -- the patient slider in DoctorUI
    only ever surfaces the single winning prototype per concept per case, so
    covering the full live set that way takes many cases.
    """

    def __init__(self, pipe, discarded=None,
                 save_path="checkpoints_deduped/discarded_prototypes.pt", thumb=130):
        self.p = pipe
        self.discarded = discarded if discarded is not None else set()
        self.save_path = save_path
        self.thumb = thumb
        self._build()

    def _build(self):
        p, hits = self.p, self.p.atlas["hit_count"]
        self.msg = widgets.HTML()
        sections = [widgets.HTML(
            "<h3>Atlas review (train-time interaction, Sec. 3.2)</h3>"
            "<p>Click a prototype to <b>discard</b> it. Discard anything whose "
            "exemplar shows a <i>shortcut</i> — ruler / scale bar, ink or marker "
            "strokes, dermoscope vignette, hair — rather than genuine skin "
            "pathology. Discards apply to every future prediction.</p>")]

        total_alive = 0
        for k in range(p.K):
            alive = [m for m in range(p.M) if int(hits[k, m]) > 0]
            total_alive += len(alive)
            cards = []
            for m in alive:
                img, img_id = p.exemplar_image(k, m)
                if img is None:
                    continue
                thumb = to_png_widget(img, width=self.thumb)
                btn = widgets.Button(
                    description=f"k={k} m={m}",
                    button_style="" if (k, m) not in self.discarded else "danger",
                    layout=widgets.Layout(width=f"{self.thumb}px"))
                btn.on_click(self._make_toggle(k, m, btn))
                cards.append(widgets.VBox(
                    [thumb, btn,
                     widgets.HTML(f"<small>{img_id}<br>{int(hits[k, m])} hits</small>")],
                    layout=widgets.Layout(margin="4px", padding="3px",
                                          border="1px solid #eee")))
            sections.append(widgets.HTML(
                f"<h4>{CONCEPT_LABELS[CONCEPT_KEYS[k]]} "
                f"&mdash; {len(alive)} alive of {p.M}</h4>"))
            sections.append(widgets.HBox(cards,
                                         layout=widgets.Layout(flex_flow="row wrap")))

        save_btn = widgets.Button(description="Save discarded prototypes",
                                  button_style="success",
                                  layout=widgets.Layout(width="230px"))
        save_btn.on_click(self.save)
        clear_btn = widgets.Button(description="Keep all (clear discards)",
                                   layout=widgets.Layout(width="200px"))
        clear_btn.on_click(self._clear)

        self.header = widgets.HTML(
            f"<b>{total_alive}</b> alive prototypes of {p.K*p.M} total "
            f"({p.K*p.M - total_alive} dead / never selected).")
        self.widget = widgets.VBox(
            [self.header, widgets.HBox([save_btn, clear_btn]), self.msg] + sections)

    def _make_toggle(self, k, m, btn):
        def handler(_):
            if (k, m) in self.discarded:
                self.discarded.discard((k, m)); btn.button_style = ""
            else:
                self.discarded.add((k, m)); btn.button_style = "danger"
            self.msg.value = f"<i>{len(self.discarded)} prototypes marked for discard</i>"
        return handler

    def _clear(self, _):
        self.discarded.clear()
        self._build()
        self.msg.value = "<i>cleared</i>"

    def save(self, _=None):
        mask = torch.ones(self.p.K, self.p.M)
        for k, m in self.discarded:
            mask[k, m] = 0.0
        torch.save({"prototype_mask": mask, "discarded": sorted(self.discarded)},
                   self.save_path)
        self.msg.value = (f"<span style='color:green'>Saved {len(self.discarded)} "
                          f"discarded prototypes &rarr; {self.save_path}</span>")
        return self.save_path


def review_atlas(source, save_path="checkpoints_deduped/discarded_prototypes.pt"):
    """
    Launch the atlas review. `source` may be a Pipeline or an existing DoctorUI
    (passing the DoctorUI shares its discard set, so both views stay in sync).
    """
    if isinstance(source, DoctorUI):
        pipe, discarded = source.p, source.discarded
    else:
        pipe, discarded = source, None
    rev = AtlasReview(pipe, discarded=discarded, save_path=save_path)
    display(rev.widget)
    return rev


def launch_ui(weights, atlas, train_img, train_task2, train_task3,
              test_img, test_task2, test_task3, backbone=DEFAULT_BACKBONE,
              device="cuda", alpha=0.3):
    pipe = load_pipeline(weights, atlas, train_img, train_task2, train_task3,
                         test_img, test_task2, test_task3, backbone, device)
    ui = DoctorUI(pipe, alpha=alpha)
    display(ui.widget)
    return ui
