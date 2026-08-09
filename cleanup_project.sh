#!/usr/bin/env bash
#
# One-time cleanup: delete superseded experiment weights and junk files.
#
# KEEPS exactly what the project needs to run and to reproduce its results:
#
#   checkpoints_revive/csr_network_final_best.pth   best static model, macro-F1 65.66
#   checkpoints_revive/phase2_best.pth              Phase-2 weights (to retrain Phase 3)
#   checkpoints_revive/atlas.pt                     concept atlas (streamlit_app, doctor_ui)
#   checkpoints_dropout/csr_network_final_best.pth  concept-dropout model (64.28 / 64.94)
#   checkpoints_p1weighted/concept_model_phase1_best.pth   Phase-1 weights (to retrain)
#   checkpoints/backbone_isic2019.pth               ISIC-2019 pretrained backbone
#
# DELETES every other checkpoint directory (failed / superseded runs), the
# regenerable experiment .pt outputs, __pycache__, and the stray '=' files.
#
# Run from the project folder:  bash cleanup_project.sh
set -u

cd "$(dirname "$0")" || exit 1
echo "Project: $(pwd)"
echo

before=$(du -sh . 2>/dev/null | cut -f1)

# --- directories to remove entirely (superseded experiments) ---------------
DIRS=(
  checkpoints_aug
  checkpoints_deduped
  checkpoints_deduped_weighted
  checkpoints_gamma100
  checkpoints_nowd
  checkpoints_p1weighted_clsw
  checkpoints_p2weighted
  checkpoints_plainCE
  checkpoints_pretrained
  checkpoints_revive_div
  __pycache__
)

# --- individual files to remove (inside directories we keep) ---------------
FILES=(
  checkpoints/concept_model_phase1_best.pth
  checkpoints/concept_model_phase1_final.pth
  checkpoints/phase2_best.pth
  checkpoints_p1weighted/csr_network_final_best.pth
  checkpoints_p1weighted/phase2_best.pth
  checkpoints_revive/impact.pt
  checkpoints_revive/confidence_delta.pt
  checkpoints_revive/concept_rejection_profiles.pt
  checkpoints_revive/selective_rejection.pt
  checkpoints_revive/mask_costly5.pt
  checkpoints_revive/mask_dominant.pt
  checkpoints_revive/mask_noop.pt
  checkpoints_revive/mask_rare_top.pt
  checkpoints_revive/mask_safe5.pt
  checkpoints_dropout/selective_rejection.pt
)

echo "The following will be PERMANENTLY DELETED:"
echo
for d in "${DIRS[@]}";  do [ -e "$d" ] && echo "  dir   $d  ($(du -sh "$d" 2>/dev/null | cut -f1))"; done
for f in "${FILES[@]}"; do [ -e "$f" ] && echo "  file  $f"; done
echo "  files =0.16 =0.9.12 =1.24 ... (empty stray files from a bad pip command)"
echo
read -r -p "Proceed? [y/N] " ans
case "$ans" in
  [yY]) ;;
  *) echo "Aborted. Nothing deleted."; exit 0 ;;
esac

echo
for d in "${DIRS[@]}";  do [ -e "$d" ] && rm -rf  "$d" && echo "removed dir   $d"; done
for f in "${FILES[@]}"; do [ -e "$f" ] && rm -f   "$f" && echo "removed file  $f"; done
rm -f '='*  2>/dev/null && echo "removed stray '=' files"

echo
echo "================= RESULT ================="
echo "Size before : $before"
echo "Size after  : $(du -sh . 2>/dev/null | cut -f1)"
echo
echo "Kept weights:"
find checkpoints checkpoints_revive checkpoints_dropout checkpoints_p1weighted \
     -type f \( -name '*.pth' -o -name '*.pt' \) 2>/dev/null \
  | while read -r f; do printf "  %-56s %s\n" "$f" "$(du -h "$f" | cut -f1)"; done
echo "=========================================="
