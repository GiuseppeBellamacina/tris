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

# Trova il comando python disponibile
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "❌ Python non trovato nel container!"
    exit 1
fi
echo "   Python: $($PY --version 2>&1)"

GPU_INFO=$($PY -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability()
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  GPU: {name} (CC {cc[0]}.{cc[1]}, {vram:.1f} GB)')
    print(f'CC_MAJOR={cc[0]}')
else:
    print('  GPU: NESSUNA GPU rilevata')
    print('CC_MAJOR=0')
" 2>&1) || true

if echo "$GPU_INFO" | grep -q "CC_MAJOR="; then
    echo "$GPU_INFO" | grep -v CC_MAJOR
    CC_MAJOR=$(echo "$GPU_INFO" | grep CC_MAJOR | cut -d= -f2)
else
    echo "  ⚠️  torch non disponibile (login node?) → assume GPU moderna (CC >= 7)"
    CC_MAJOR=8
fi

# ── 2. Installa dipendenze dal pyproject.toml ─────────────────────────────────
echo ""
if [ "$CC_MAJOR" -ge 7 ] 2>/dev/null; then
    echo "📦 GPU CC >= 7.0 → installazione completa (base + fast)..."
    pip install --user -e ".[fast]"
else
    echo "📦 GPU CC < 7.0 → installazione base (senza Unsloth/vLLM)..."
    echo "   Usa config con: use_unsloth: false, fast_inference: false"
    pip install --user -e .
fi

# ── 3. Genera dataset sintetico ───────────────────────────────────────────────
if [ ! -d "data/synthetic" ]; then
    echo ""
    echo "📊 Generazione dataset sintetico (5000 samples)..."
    $PY -c "
from src.datasets.synthetic_dataset import generate_dataset
ds = generate_dataset(num_samples=5000, seed=42)
ds.save_to_disk('data/synthetic')
print('Dataset salvato in data/synthetic')
"
else
    echo ""
    echo "✅ Dataset già presente in data/synthetic"
fi

# ── 4. Verifica installazione ─────────────────────────────────────────────────
echo ""
echo "🔍 Verifica installazione..."
$PY -c "
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
