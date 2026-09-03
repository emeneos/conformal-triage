#!/usr/bin/env python3
# 02_extract_embeddings.py -- Phase 2: run frozen PanDerm ONCE over each source and
# cache the embeddings. After this the GPU is never touched again by the main pipeline.
#
# Run once per source (set ROOT to the folder used by 01_download.sh):
#   ROOT=$HOME/conformal-triage-data
#   python 02_extract_embeddings.py --panderm $ROOT/panderm/PanDerm \
#       --ckpt $ROOT/panderm/checkpoints/panderm_ll_data6_checkpoint-499.pth \
#       --images $ROOT/isic2019/ISIC_2019_Training_Input --out $ROOT/emb/isic2019.npz
#   python 02_extract_embeddings.py --panderm ... --ckpt ... \
#       --images $ROOT/hiba/images --out $ROOT/emb/hiba.npz
#   python 02_extract_embeddings.py --panderm ... --ckpt ... \
#       --images $ROOT/pad/images --out $ROOT/emb/pad.npz
#
# Output: an npz with ids (file stems) and features (N, d) float32, plus a .json sidecar
# with checkpoint, count and dimension, for traceability.

import argparse, json, sys, time
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--panderm", required=True, help="path to the cloned PanDerm/ repository")
p.add_argument("--ckpt", required=True, help="checkpoint panderm_ll_data6_checkpoint-499.pth")
p.add_argument("--images", required=True, help="folder with the images (recursive)")
p.add_argument("--out", required=True, help="output npz")
p.add_argument("--model-name", default="PanDerm_Large_LP")
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--num-workers", type=int, default=4)
p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
p.add_argument("--amp", action="store_true", help="fp16 autocast on GPU (faster, same result for practical purposes)")
args = p.parse_args()

# the official API lives in PanDerm/classification (from models import get_encoder)
sys.path.insert(0, str(Path(args.panderm) / "classification"))

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from models import get_encoder  # noqa: E402  (PanDerm repository)

device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

class CkptArgs:  # get_encoder expects an object with this attribute
    pretrained_checkpoint = args.ckpt

model, eval_transform = get_encoder(CkptArgs(), model_name=args.model_name)
model.eval().to(device)

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
paths = sorted(q for q in Path(args.images).rglob("*") if q.suffix.lower() in EXTS)
if not paths:
    sys.exit(f"No images found in {args.images}")
print(f"{len(paths)} images | device={device} | model={args.model_name}")

class ImgDS(Dataset):
    def __len__(self):
        return len(paths)
    def __getitem__(self, i):
        q = paths[i]
        try:
            img = Image.open(q).convert("RGB")
        except Exception as e:  # corrupt image: do not abort the whole run
            print(f"  ! error reading {q.name}: {e}; using a blank image", file=sys.stderr)
            img = Image.new("RGB", (224, 224))
        return eval_transform(img), q.stem

dl = DataLoader(ImgDS(), batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=(device == "cuda"))

feats, ids, t0 = [], [], time.time()
autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if (args.amp and device == "cuda") \
    else torch.autocast(device_type="cpu", enabled=False)
with torch.no_grad():
    for bi, (x, stems) in enumerate(dl):
        x = x.to(device, non_blocking=True)
        with autocast:
            f = model.forward_features(x, is_train=False)
        feats.append(f.float().cpu().numpy())
        ids.extend(stems)
        if bi % 50 == 0:
            done = (bi + 1) * args.batch_size
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {min(done, len(paths))}/{len(paths)}  ({rate:.1f} img/s)", flush=True)

F = np.concatenate(feats, axis=0).astype(np.float32)
out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, ids=np.array(ids), features=F)
meta = {"source_dir": str(args.images), "checkpoint": str(args.ckpt),
        "model_name": args.model_name, "n_images": int(F.shape[0]),
        "dim": int(F.shape[1]), "device": device,
        "date": time.strftime("%Y-%m-%d %H:%M:%S")}
out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
print(f"OK -> {out}  shape={F.shape}")
