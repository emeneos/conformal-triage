#!/usr/bin/env bash
# 01_download.sh -- Fase 1 del roadmap: datos + modelo.
# Uso:  bash 01_download.sh [carpeta_destino]     (default: ~/conformal-triage-data)
# Reanudable: wget -c y unzip -n permiten cortar y volver a correr sin re-bajar.
set -uo pipefail

ROOT="${1:-$HOME/conformal-triage-data}"
mkdir -p "$ROOT"/{isic2019,hiba,pad,panderm/checkpoints}
echo ">> destino: $ROOT"

# ---------------------------------------------------------------- ISIC 2019
# Links oficiales del challenge (verificados en challenge.isic-archive.com/data).
echo ">> [1/4] ISIC 2019 (~9.1 GB)"
cd "$ROOT/isic2019"
wget -c https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip
wget -c https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv
wget -c https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Metadata.csv
unzip -qn ISIC_2019_Training_Input.zip

# ---------------------------------------------------------------- HIBA
# Coleccion "Hospital Italiano de Buenos Aires - Skin Lesions Images (2019-2022)"
# en el ISIC Archive. El ID se busca, no se hardcodea (puede cambiar).
echo ">> [2/4] HIBA via isic-cli"
python3 -m pip install -q isic-cli
cd "$ROOT/hiba"
HIBA_ID=$(isic collection list 2>/dev/null | grep -i "hospital italiano" | grep -oE "[0-9]+" | head -1)
if [ -z "${HIBA_ID:-}" ]; then
  echo "!! No encontre la coleccion automaticamente."
  echo "   Corre 'isic collection list', busca 'Hospital Italiano de Buenos Aires'"
  echo "   y despues:  isic image download --collections <ID> $ROOT/hiba/images/"
else
  echo "   coleccion HIBA id=$HIBA_ID"
  isic metadata download --collections "$HIBA_ID" || true   # guarda tambien la metadata
  isic image download --collections "$HIBA_ID" images/
fi

# ---------------------------------------------------------------- PAD-UFES-20
# Mirror oficial en el ISIC Archive (coleccion "PAD-UFES-20"); el link directo de
# Mendeley dejo de funcionar, y el mirror ademas trae la metadata en el mismo
# esquema que HIBA (diagnosis_1/2/3, fototipo, lesion_id/patient_id).
echo ">> [3/4] PAD-UFES-20 via isic-cli"
cd "$ROOT/pad"
PAD_ID=$(isic collection list 2>/dev/null | grep -i "pad-ufes" | grep -oE "[0-9]+" | head -1)
if [ -z "${PAD_ID:-}" ]; then
  echo "!! No encontre la coleccion. Corre 'isic collection list', busca 'PAD-UFES-20'"
  echo "   y despues: isic image download --collections <ID> $ROOT/pad/images/"
else
  echo "   coleccion PAD-UFES-20 id=$PAD_ID"
  isic metadata download --collections "$PAD_ID" || true
  isic image download --collections "$PAD_ID" images/
fi

# ---------------------------------------------------------------- PanDerm
echo ">> [4/4] PanDerm (repo + checkpoint ViT-L)"
cd "$ROOT/panderm"
[ -d PanDerm ] || git clone https://github.com/SiyuanYan1/PanDerm
python3 -m pip install -q gdown
CKPT="checkpoints/panderm_ll_data6_checkpoint-499.pth"
if [ ! -f "$CKPT" ]; then
  # ID del Google Drive publicado en el README del repo (modelo del paper, ViT-L/16)
  gdown 1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE -O "$CKPT" || {
    echo "!! gdown fallo. Baja el checkpoint a mano desde el link 'PanDerm' del README:"
    echo "   https://github.com/SiyuanYan1/PanDerm#1-download-panderm-pre-trained-weights"
    echo "   y guardalo como $ROOT/panderm/$CKPT"; }
fi

# ---------------------------------------------------------------- sanidad
echo; echo "== conteo de sanidad (esperado: 25331 / 1616 / 2298) =="
echo "ISIC 2019: $(find "$ROOT/isic2019" -name '*.jpg' | wc -l) jpg"
echo "HIBA:      $(find "$ROOT/hiba/images" -type f 2>/dev/null | wc -l) imagenes"
echo "PAD:       $(find "$ROOT/pad/images" -type f 2>/dev/null | wc -l) imagenes"
echo "PanDerm:   $(ls -lh "$ROOT/panderm/$CKPT" 2>/dev/null | awk '{print $5}' ) checkpoint"
echo "Listo. Siguiente paso: 02_extract_embeddings.py (comandos en su header)."
