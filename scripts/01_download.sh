#!/usr/bin/env bash
# 01_download.sh -- Phase 1: data + model (local, non-Colab alternative to notebook 01).
# Usage:  bash 01_download.sh [target_dir]     (default: ~/conformal-triage-data)
# Resumable: wget -c and unzip -n allow stopping and re-running without re-downloading.
set -uo pipefail

ROOT="${1:-$HOME/conformal-triage-data}"
mkdir -p "$ROOT"/{isic2019,hiba,pad,panderm/checkpoints}
echo ">> target: $ROOT"

# ---------------------------------------------------------------- ISIC 2019
# Official challenge links (verified at challenge.isic-archive.com/data).
echo ">> [1/4] ISIC 2019 (~9.1 GB)"
cd "$ROOT/isic2019"
wget -c https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip
wget -c https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv
wget -c https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Metadata.csv
unzip -qn ISIC_2019_Training_Input.zip

# ---------------------------------------------------------------- HIBA
# Collection "Hospital Italiano de Buenos Aires - Skin Lesions Images (2019-2022)"
# in the ISIC Archive. The ID is looked up, not hard-coded (it may change).
echo ">> [2/4] HIBA via isic-cli"
python3 -m pip install -q isic-cli
cd "$ROOT/hiba"
HIBA_ID=$(isic collection list 2>/dev/null | grep -i "hospital italiano" | grep -oE "[0-9]+" | head -1)
if [ -z "${HIBA_ID:-}" ]; then
  echo "!! Could not find the collection automatically."
  echo "   Run 'isic collection list', look for 'Hospital Italiano de Buenos Aires'"
  echo "   and then:  isic image download --collections <ID> $ROOT/hiba/images/"
else
  echo "   HIBA collection id=$HIBA_ID"
  isic metadata download --collections "$HIBA_ID" || true   # also saves the metadata
  isic image download --collections "$HIBA_ID" images/
fi

# ---------------------------------------------------------------- PAD-UFES-20
# Official mirror in the ISIC Archive (collection "PAD-UFES-20"); the direct Mendeley
# link stopped working, and the mirror also ships the metadata in the same schema as
# HIBA (diagnosis_1/2/3, phototype, lesion_id/patient_id).
echo ">> [3/4] PAD-UFES-20 via isic-cli"
cd "$ROOT/pad"
PAD_ID=$(isic collection list 2>/dev/null | grep -i "pad-ufes" | grep -oE "[0-9]+" | head -1)
if [ -z "${PAD_ID:-}" ]; then
  echo "!! Could not find the collection. Run 'isic collection list', look for 'PAD-UFES-20'"
  echo "   and then: isic image download --collections <ID> $ROOT/pad/images/"
else
  echo "   PAD-UFES-20 collection id=$PAD_ID"
  isic metadata download --collections "$PAD_ID" || true
  isic image download --collections "$PAD_ID" images/
fi

# ---------------------------------------------------------------- PanDerm
echo ">> [4/4] PanDerm (repo + ViT-L checkpoint)"
cd "$ROOT/panderm"
[ -d PanDerm ] || git clone https://github.com/SiyuanYan1/PanDerm
python3 -m pip install -q gdown
CKPT="checkpoints/panderm_ll_data6_checkpoint-499.pth"
if [ ! -f "$CKPT" ]; then
  # Google Drive ID published in the repository README (paper model, ViT-L/16)
  gdown 1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE -O "$CKPT" || {
    echo "!! gdown failed. Download the checkpoint manually from the 'PanDerm' link in the README:"
    echo "   https://github.com/SiyuanYan1/PanDerm#1-download-panderm-pre-trained-weights"
    echo "   and save it as $ROOT/panderm/$CKPT"; }
fi

# ---------------------------------------------------------------- sanity
echo; echo "== sanity counts (expected: 25331 / 1616 / 2298) =="
echo "ISIC 2019: $(find "$ROOT/isic2019" -name '*.jpg' | wc -l) jpg"
echo "HIBA:      $(find "$ROOT/hiba/images" -type f 2>/dev/null | wc -l) images"
echo "PAD:       $(find "$ROOT/pad/images" -type f 2>/dev/null | wc -l) images"
echo "PanDerm:   $(ls -lh "$ROOT/panderm/$CKPT" 2>/dev/null | awk '{print $5}' ) checkpoint"
echo "Done. Next step: 02_extract_embeddings.py (commands in its header)."
