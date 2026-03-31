#!/bin/bash
# ============================================================================
# Setup one-tantum per il cluster DMI.
#
# Eseguire UNA VOLTA dentro una sessione interattiva Apptainer:
#
#   srun --account <QUEUE> --partition <QUEUE> --qos gpu-small \
#        --gres=gpu:1 --gres=shard:2000 --mem=4G \
#        --pty apptainer shell --nv /shared/sifs/latest.sif
#
#   # dentro il container:
#   cd ~/GRPO-strict-generation
#   bash cluster/setup.sh
# ============================================================================
set -e

echo "=== Setup GRPO Strict Generation (Cluster DMI) ==="
echo ""

# ── 1. Verifica GPU ──────────────────────────────────────────────────────────
echo "🔍 Rilevamento GPU..."
GPU_INFO=$(python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability()
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'{name} (CC {cc[0]}.{cc[1]}, {vram:.1f} GB)')
    print(f'CC_MAJOR={cc[0]}')
else:
    print('NESSUNA GPU')
    print('CC_MAJOR=0')
" 2>/dev/null)

echo "   GPU: $(echo "$GPU_INFO" | head -1)"
CC_MAJOR=$(echo "$GPU_INFO" | grep CC_MAJOR | cut -d= -f2)

# ── 2. Installa dipendenze base ──────────────────────────────────────────────
echo ""
echo "📦 Installazione dipendenze base (pip --user)..."
pip install --user --quiet \
    trl==0.24.0 peft bitsandbytes accelerate datasets \
    wandb seaborn scikit-learn python-dotenv tqdm pyyaml \
    matplotlib numpy pandas ipywidgets tensorboard

# ── 3. Installa Unsloth + vLLM se GPU supportata (CC >= 7) ───────────────────
if [ "$CC_MAJOR" -ge 7 ] 2>/dev/null; then
    echo ""
    echo "📦 GPU CC >= 7.0 → installazione Unsloth + vLLM..."
    pip install --user --quiet unsloth==2026.3.17 vllm==0.18.0
else
    echo ""
    echo "⚠️  GPU CC < 7.0 (K80?) → Unsloth e vLLM NON supportati."
    echo "   Usa config con: use_unsloth: false, fast_inference: false"
fi

# ── 4. Installa il progetto in editable mode ──────────────────────────────────
echo ""
echo "📦 Installazione progetto (editable)..."
pip install --user --no-deps -e .

# ── 5. Genera dataset sintetico ───────────────────────────────────────────────
if [ ! -d "data/synthetic" ]; then
    echo ""
    echo "📊 Generazione dataset sintetico (5000 samples)..."
    python3 -c "
from src.datasets.synthetic_dataset import generate_dataset
ds = generate_dataset(num_samples=5000, seed=42)
ds.save_to_disk('data/synthetic')
print('Dataset salvato in data/synthetic')
"
else
    echo ""
    echo "✅ Dataset già presente in data/synthetic"
fi

# ── 6. Verifica installazione ─────────────────────────────────────────────────
echo ""
echo "🔍 Verifica installazione..."
python3 -c "
import torch, transformers, trl, peft, datasets
print(f'  PyTorch:       {torch.__version__}')
print(f'  Transformers:  {transformers.__version__}')
print(f'  TRL:           {trl.__version__}')
print(f'  PEFT:          {peft.__version__}')
print(f'  Datasets:      {datasets.__version__}')
try:
    import unsloth
    print(f'  Unsloth:       {unsloth.__version__}')
except ImportError:
    print(f'  Unsloth:       NON installato (GPU non supportata)')
try:
    import vllm
    print(f'  vLLM:          {vllm.__version__}')
except ImportError:
    print(f'  vLLM:          NON installato')
try:
    import bitsandbytes
    print(f'  bitsandbytes:  {bitsandbytes.__version__}')
except ImportError:
    print(f'  bitsandbytes:  NON installato')
"

echo ""
echo "=== ✅ Setup completato! ==="
echo ""
echo "Prossimi passi:"
echo "  1. Modifica cluster/train.sh con la tua queue, email e QoS"
echo "  2. Lancia: sbatch cluster/train.sh"
