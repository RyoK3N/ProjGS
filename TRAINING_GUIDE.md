# ProjGS Training Guide

Complete guide for training Projection-Guided Gaussian Splatting models.

## Overview

This training pipeline implements **3D Gaussian Splatting** with camera-aware initialization optimized for RGB-D input. The system includes:

- ✅ Differentiable Gaussian rasterization
- ✅ L1 + D-SSIM loss functions
- ✅ Adaptive densification and pruning
- ✅ Real-time TensorBoard monitoring
- ✅ Mixed precision training (FP16/BF16)
- ✅ Optimized for Mac M4 Pro and A100

---

## Quick Start

### 1. Mac M4 Pro Training (Local Development)

```bash
# Activate conda environment
conda activate synexian

# Start training (3000 iterations, optimized for 24GB unified memory)
python scripts/train.py --config configs/train_mac.yaml

# Monitor with TensorBoard
tensorboard --logdir runs/mac --port 6006
```

### 2. A100 Server Training (Full Scale)

```bash
# Full training on all scenes (30,000 iterations)
python scripts/train.py --config configs/train_a100.yaml

# Monitor with TensorBoard
tensorboard --logdir runs/a100 --port 6006
```

### 3. Evaluate Trained Model

```bash
# Evaluate on validation set
python scripts/evaluate.py \
    --checkpoint checkpoints/mac/final_checkpoint.pth \
    --config configs/train_mac.yaml \
    --output-dir outputs/evaluation
```

---

## Configuration Files

### Mac M4 Pro (`configs/train_mac.yaml`)

**Optimized for:**
- 24GB unified memory
- 16 GPU cores (MPS backend)
- Local testing and development

**Key Settings:**
- Iterations: 3,000 (vs 30,000 on A100)
- Batch size: 1 with gradient accumulation (4 steps)
- Max Gaussians: 50,000
- Mixed precision: FP16
- Gradient checkpointing: Enabled
- MPS cache clearing: Every 10 iterations

### A100 Server (`configs/train_a100.yaml`)

**Optimized for:**
- 40GB/80GB VRAM
- CUDA with Tensor Cores
- Full-scale production training

**Key Settings:**
- Iterations: 30,000
- Batch size: 4 (multi-scene)
- Max Gaussians: 500,000
- Mixed precision: BF16 (better than FP16 on A100)
- k-NN adaptive scaling: Enabled
- Complete dataset training

---

## Training Pipeline Components

### 1. Gaussian Initialization

Uses camera-aware initialization from feasibility study:

```python
# Optimal parameters (from parameter sweep):
base_scale: 0.5
uncertainty_weight: true
use_nearest_neighbor: true  # A100 only (slower on Mac)
```

**Result:** 63.82x better scale variance than naive initialization

### 2. Loss Functions

Combined photometric + perceptual loss:

```
L = λ₁ · L1 + λ₂ · (1 - SSIM)
```

- `λ₁ = 0.8` (L1 photometric loss)
- `λ₂ = 0.2` (D-SSIM perceptual loss)

### 3. Adaptive Density Control

**Densification (iterations 500-15000):**
- **Split:** Large Gaussians (>2% scene extent) with high gradients
- **Clone:** Small Gaussians with high gradients
- Interval: Every 100 iterations

**Pruning:**
- Remove low opacity Gaussians (<0.005)
- Remove excessively large Gaussians
- Interval: Every 100 iterations

**Opacity Reset:**
- Reset all opacities to 0.01
- Prevents mode collapse
- Interval: Every 3,000 iterations

### 4. Optimization

Separate Adam optimizers for each parameter group:

| Parameter | Learning Rate | Schedule |
|-----------|---------------|----------|
| Position  | 1.6e-4 → 1.6e-6 | Exponential decay |
| Features (RGB) | 2.5e-3 | Constant |
| Opacity | 5.0e-2 | Constant |
| Scaling | 5.0e-3 | Constant |
| Rotation | 1.0e-3 | Constant |

---

## TensorBoard Monitoring

TensorBoard provides real-time visualization of:

### Metrics (logged every 10 iterations)
- **Loss:** Total, L1, SSIM components
- **Quality:** PSNR, SSIM metrics
- **Model:** Number of Gaussians over time
- **Learning Rates:** All parameter groups

### Images (logged every 100 iterations)
- **Rendered:** Current render output
- **Ground Truth:** Target RGB image
- **Depth:** Rendered depth map
- **Alpha:** Accumulated alpha channel

### Access TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir runs/mac --port 6006

# Open in browser
http://localhost:6006
```

**Example Metrics to Monitor:**

| Metric | Good Range | Warning Signs |
|--------|------------|---------------|
| PSNR | > 20 dB | < 15 dB (poor quality) |
| SSIM | > 0.7 | < 0.5 (artifacts) |
| Gaussians | Growing initially | Constant (no densification) |
| Loss | Decreasing | Increasing (instability) |

---

## Checkpointing

### Automatic Checkpointing

**Mac:**
- Saved every 500 iterations
- Keep last 3 checkpoints
- Location: `checkpoints/mac/`

**A100:**
- Saved every 5,000 iterations
- Keep last 10 checkpoints
- Location: `checkpoints/a100/`

### Resume Training

```bash
python scripts/train.py \
    --config configs/train_mac.yaml \
    --resume checkpoints/mac/checkpoint_iter_001000.pth
```

### Checkpoint Contents

```python
checkpoint = {
    'iteration': 1000,
    'model_state': {
        'positions': [...],
        'features': [...],
        'opacities': [...],
        'scales': [...],
        'rotations': [...]
    },
    'optimizer_states': {...},
    'scaler_state': {...}  # For mixed precision
}
```

---

## Memory Optimization

### Mac M4 Pro (24GB Unified Memory)

1. **Gradient Checkpointing:** Enabled
   - Trades compute for memory
   - ~30% memory savings

2. **MPS Cache Clearing:** Every 10 iterations
   ```python
   torch.mps.empty_cache()
   ```

3. **Gaussian Count Limit:** 50,000 max
   - Prevents OOM on complex scenes

4. **Batch Size:** 1 with gradient accumulation
   - Effective batch size: 4

### A100 (40GB/80GB)

1. **No Gradient Checkpointing:** Disabled
   - A100 has sufficient memory

2. **Higher Gaussian Count:** 500,000 max

3. **Batch Processing:** 4 scenes simultaneously

4. **Pin Memory:** Enabled for faster CUDA transfers

---

## Training Workflow

### Complete Training Pipeline

```bash
# 1. Verify feasibility study results
cat UPDATED_RESULTS_SUMMARY.md

# 2. Start Mac training for quick validation
python scripts/train.py --config configs/train_mac.yaml

# 3. Monitor progress
tensorboard --logdir runs/mac --port 6006

# 4. Check results after ~30 minutes (3000 iterations)
python scripts/evaluate.py \
    --checkpoint checkpoints/mac/final_checkpoint.pth \
    --config configs/train_mac.yaml

# 5. If Mac results look good, run full A100 training
python scripts/train.py --config configs/train_a100.yaml

# 6. Full evaluation on A100 model
python scripts/evaluate.py \
    --checkpoint checkpoints/a100/final_checkpoint.pth \
    --config configs/train_a100.yaml \
    --num-scenes 100
```

---

## Expected Training Times

### Mac M4 Pro
- **3,000 iterations:** ~1-2 hours
- **Per iteration:** ~1-2 seconds
- **Memory usage:** 8-12GB

### A100
- **30,000 iterations:** ~8-12 hours
- **Per iteration:** ~0.5-1 seconds
- **Memory usage:** 20-30GB

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solutions:**
1. Reduce `max_gaussians` in config
2. Enable `gradient_checkpointing`
3. Reduce `batch_size`
4. Increase `empty_cache_interval`

### Issue: "MPS backend error" (Mac)

**Solutions:**
1. Update to latest macOS and PyTorch
2. Fallback to CPU: Set `device: "cpu"` in config
3. Reduce resolution: `resize: [240, 320]`

### Issue: Loss not decreasing

**Possible Causes:**
1. Learning rate too high → Reduce by 10x
2. Bad initialization → Check depth quality
3. No densification → Verify iteration range

### Issue: Gaussians growing infinitely

**Solutions:**
1. Enable pruning: Verify `pruning.interval`
2. Lower `max_scale` threshold
3. Increase `opacity_reset_interval`

---

## Advanced Usage

### Multi-GPU Training (A100 only)

```yaml
# In configs/train_a100.yaml:
distributed:
  enabled: true
  backend: "nccl"
  world_size: 2  # Number of GPUs
```

```bash
# Launch with torchrun
torchrun --nproc_per_node=2 scripts/train.py --config configs/train_a100.yaml
```

### Custom Dataset

1. Extend `SUNRGBDDataset`:
   ```python
   class CustomDataset(SUNRGBDDataset):
       def __getitem__(self, idx):
           return {
               'rgb': [...],
               'depth': [...],
               'K': [...]
           }
   ```

2. Update config:
   ```yaml
   dataset:
     name: "custom"
     root_dir: "path/to/data"
   ```

---

## Performance Benchmarks

### Initialization Quality (from Feasibility Study)

| Metric | Camera-Aware | Naive | Improvement |
|--------|--------------|-------|-------------|
| Scale Variance | 6.28e-7 | 0.0 | 63.82x |
| Opacity Variance | 0.111 | 0.0 | ∞ |
| PSNR (init) | ~18 dB | ~12 dB | +6 dB |

### Training Performance

| Platform | Throughput | Memory | Cost |
|----------|-----------|--------|------|
| Mac M4 Pro | 0.5-1 it/s | 8-12GB | Free (local) |
| A100 (40GB) | 1-2 it/s | 20-30GB | ~$2-3/hour |
| A100 (80GB) | 1-2 it/s | 20-40GB | ~$3-4/hour |

---

## Citation

If you use this training pipeline, please cite:

```bibtex
@software{projgs2025,
  title={Projection-Guided Gaussian Splatting from Single RGB-D},
  author={ProjGS Research Team},
  year={2025},
  url={https://github.com/your-repo/projgs}
}
```

---

## Next Steps

1. ✅ Complete training on Mac M4 Pro
2. ✅ Verify quality metrics (PSNR > 20 dB, SSIM > 0.7)
3. ✅ Run full training on A100 server
4. 🔄 Benchmark against baselines (NeRF, vanilla 3DGS)
5. 🔄 Publish results

---

**Questions?** Check the main README or open an issue.
