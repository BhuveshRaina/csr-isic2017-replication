#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Download the official ISIC-2017 validation + test archives (images + concept
# ground truth + diagnosis labels) used to reproduce Table 1 of the paper.
#
# The paper's reported ISIC macro-F1 (71.5) is measured on the official
# hold-out TEST set (600 images). Your project already contains the TRAINING
# split. This script fetches Validation (150) and Test v2 (600).
#
# Usage:   bash download_isic2017.sh [target_dir]
# Default target_dir = ./data
# -----------------------------------------------------------------------------
set -euo pipefail

DEST="${1:-./data}"
BASE="https://isic-archive.s3.amazonaws.com/challenges/2017"
mkdir -p "$DEST"
cd "$DEST"

echo "Downloading ISIC-2017 validation + test archives into: $(pwd)"

files=(
  # --- Validation (150 imgs) ---
  "ISIC-2017_Validation_Data.zip"                 # images (+ superpixel PNGs)
  "ISIC-2017_Validation_Part2_GroundTruth.zip"    # concept JSONs
  "ISIC-2017_Validation_Part3_GroundTruth.csv"    # diagnosis labels
  # --- Test v2 (600 imgs, the paper's reporting set) ---
  "ISIC-2017_Test_v2_Data.zip"                    # images (+ superpixel PNGs) ~5.4 GB
  "ISIC-2017_Test_v2_Part2_GroundTruth.zip"       # concept JSONs
  "ISIC-2017_Test_v2_Part3_GroundTruth.csv"       # diagnosis labels
)

for f in "${files[@]}"; do
  if [[ -f "$f" ]]; then
    echo "  [skip] $f already exists"
  else
    echo "  [get ] $f"
    curl -L -C - -o "$f" "$BASE/$f"
  fi
done

echo
echo "Done. You now have (in $DEST):"
echo "  ISIC-2017_Test_v2_Data.zip / _Part2_GroundTruth.zip / _Part3_GroundTruth.csv"
echo "  ISIC-2017_Validation_Data.zip / _Part2_GroundTruth.zip / _Part3_GroundTruth.csv"
echo
echo "Optional (paper pretrains the ISIC baseline on ISIC-2019, 25k imgs):"
echo "  $BASE/../2019/ISIC_2019_Training_Input.zip   (9.1 GB)"
