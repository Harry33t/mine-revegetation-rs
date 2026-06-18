#!/bin/bash
# Run ON the AutoDL GPU box after the repo + tiles are uploaded to
# /root/autodl-tmp/mine-revegetation-rs-2. AutoDL images ship torch preinstalled,
# so we only add transformers + smp (HF behind GFW -> hf-mirror; pip -> aliyun).
set -e
REPO=/root/autodl-tmp/mine-revegetation-rs-2
cd "$REPO"

export HF_ENDPOINT=https://hf-mirror.com           # HuggingFace weights mirror
PIP="pip install -q -i https://mirrors.aliyun.com/pypi/simple/"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || { echo 'torch missing in this env'; exit 1; }

$PIP "transformers>=4.40" "segmentation-models-pytorch>=0.3.3"
python -c "import transformers, segmentation_models_pytorch; print('deps OK')"

echo "Setup done. Train with e.g.:"
echo "  PYTHONPATH=src HF_ENDPOINT=$HF_ENDPOINT python -m reveg.models.train \\"
echo "    --sites data/processed/tiles/alcoa_huntly data/processed/tiles/ranger data/processed/tiles/mt_owen \\"
echo "    --model segformer --fractions 0.01,0.05,0.1,1.0 --epochs 40 --out outputs/seg"
