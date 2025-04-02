torchrun --nnodes 1 --nproc-per-node 2 norm_layer.py \
  --launcher pytorch --work-dir runs/bs60-bn \
  --num-workers 16 --batch-size 30 --norm BatchNorm \
  --warm-up-ratio 0.1 --warn-up-steps 4.0 \
  --max-epochs 30 --step-size 50 --ema-epoch 24

torchrun --nnodes 1 --nproc-per-node 2 norm_layer.py \
  --launcher pytorch --work-dir runs/bs60-ln \
  --num-workers 16 --batch-size 30 --norm LayerNorm \
  --warm-up-ratio 0.1 --warn-up-steps 4.0 \
  --max-epochs 30 --step-size 50 --ema-epoch 24

torchrun --nnodes 1 --nproc-per-node 2 norm_layer.py \
  --launcher pytorch --work-dir runs/bs120-bn \
  --num-workers 16 --batch-size 60 --norm BatchNorm \
  --warm-up-ratio 0.1 --warn-up-steps 4.0 \
  --max-epochs 30 --step-size 50 --ema-epoch 24

torchrun --nnodes 1 --nproc-per-node 2 norm_layer.py \
  --launcher pytorch --work-dir runs/bs120-ln \
  --num-workers 16 --batch-size 60 --norm LayerNorm \
  --warm-up-ratio 0.1 --warn-up-steps 4.0 \
  --max-epochs 30 --step-size 50 --ema-epoch 24

torchrun --nnodes 1 --nproc-per-node 2 norm_layer.py \
  --launcher pytorch --work-dir runs/bs180-bn \
  --num-workers 16 --batch-size 90 --norm BatchNorm \
  --warm-up-ratio 0.1 --warn-up-steps 4.0 \
  --max-epochs 30 --step-size 50 --ema-epoch 24

torchrun --nnodes 1 --nproc-per-node 2 norm_layer.py \
  --launcher pytorch --work-dir runs/bs180-ln \
  --num-workers 16 --batch-size 90 --norm LayerNorm \
  --warm-up-ratio 0.1 --warn-up-steps 4.0 \
  --max-epochs 30 --step-size 50 --ema-epoch 24
