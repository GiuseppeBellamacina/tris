#!/bin/bash
# ============================================================================
# Lancia training + evaluation per più modelli in catena.
#
# La QoS permette un solo job alla volta, quindi un watcher in background
# controlla ogni 60s se la coda è vuota e sottomette il prossimo job.
#
# Uso:
#   bash cluster/run_all.sh                       # tutti i modelli, train+eval
#   bash cluster/run_all.sh --eval-only            # solo evaluation (skip training)
#   bash cluster/run_all.sh --train-only           # solo training (skip eval)
#   bash cluster/run_all.sh --models=1,3,5         # solo modelli 1, 3 e 5
#   bash cluster/run_all.sh --models=1t,2e,3       # 1=solo train, 2=solo eval, 3=entrambi
#   bash cluster/run_all.sh --models=4t,5te        # 4=solo train, 5=train+eval
#
# Suffissi per --models:
#   (nessuno)  train + eval (default)
#   t          solo train
#   e          solo eval
#   te / et    train + eval (esplicito)
#
# Monitorare:
#   tail -f logs/chain_watcher.log           # log del watcher
#   ps aux | grep chain_next | grep -v grep  # vedi se il watcher è attivo
#   myjobs                                   # job attivo su SLURM
#
# Interrompere:
#   kill $(cat .chain_pid)                   # uccidi il watcher
#   killalljobs                              # cancella anche il job SLURM attivo
#
# Il watcher si chiude da solo quando la catena è completata.
#
# Ogni modello ha la sua config con output_dir e log_dir separati,
# quindi non ci sono conflitti tra i risultati.
# ============================================================================

set -e

# ── Parsing argomenti ─────────────────────────────────────────────────────────
GLOBAL_TRAIN=1
GLOBAL_EVAL=1
ONLY_MODELS=""
for arg in "$@"; do
    case "$arg" in
        --eval-only)  GLOBAL_TRAIN=0 ;;
        --train-only) GLOBAL_EVAL=0 ;;
        --models=*)   ONLY_MODELS="${arg#--models=}" ;;
        --help|-h)
            echo "Uso: bash cluster/run_all.sh [--eval-only] [--train-only] [--models=SPEC]"
            echo ""
            echo "Opzioni globali:"
            echo "  --eval-only      Solo evaluation per tutti (skip training)"
            echo "  --train-only     Solo training per tutti (skip eval)"
            echo ""
            echo "Selezione modelli (--models=SPEC):"
            echo "  SPEC è una lista separata da virgole: INDICE[SUFFISSO],..."
            echo "  Suffissi:  t = solo train,  e = solo eval,  te = entrambi (default)"
            echo ""
            echo "  Esempi:"
            echo "    --models=1,3,5       modelli 1,3,5 con train+eval"
            echo "    --models=1t,2e,3     1=solo train, 2=solo eval, 3=entrambi"
            echo "    --models=4t,5te      4=solo train, 5=train+eval"
            echo ""
            echo "Modelli disponibili:"
            echo "  1: smollm2-135m"
            echo "  2: smollm2-360m"
            echo "  3: qwen25-05b"
            echo "  4: tinyllama-11b"
            echo "  5: gemma2-2b"
            exit 0
            ;;
    esac
done

# ── Modelli da lanciare ───────────────────────────────────────────────────────
# Formato: "TAG:CONFIG_PATH"
MODELS=(
    "smollm2-135m:experiments/configs/grpo_smollm2_135m.yaml"
    "smollm2-360m:experiments/configs/grpo_smollm2_360m.yaml"
    "qwen25-05b:experiments/configs/grpo_qwen05.yaml"
    "tinyllama-11b:experiments/configs/grpo_tinyllama.yaml"
    "gemma2-2b:experiments/configs/grpo_gemma2.yaml"
)

PROJ_DIR="$HOME/GRPO-strict-generation"
CHAIN_FILE="$PROJ_DIR/.job_chain"

# ── Filtra modelli se --models è specificato ──────────────────────────────────
# Ogni entry diventa "TAG:CONFIG:do_train:do_eval"
SELECTED=()

if [ -n "$ONLY_MODELS" ]; then
    IFS=',' read -ra SPECS <<< "$ONLY_MODELS"
    for spec in "${SPECS[@]}"; do
        # Separa indice numerico e suffisso lettere: "3te" → idx=3, suffix="te"
        idx=$(echo "$spec" | grep -oP '^\d+')
        suffix=$(echo "$spec" | grep -oP '[a-zA-Z]+$' || true)
        i=$((idx - 1))  # 1-based → 0-based

        if [ $i -lt 0 ] || [ $i -ge ${#MODELS[@]} ]; then
            echo "⚠️  Indice $idx fuori range (1-${#MODELS[@]})"
            continue
        fi

        # Determina train/eval per questo modello
        case "$suffix" in
            t)   do_t=1; do_e=0 ;;
            e)   do_t=0; do_e=1 ;;
            te|et) do_t=1; do_e=1 ;;
            "")  do_t=$GLOBAL_TRAIN; do_e=$GLOBAL_EVAL ;;
            *)   echo "⚠️  Suffisso sconosciuto '$suffix' per modello $idx (usa t/e/te)"; continue ;;
        esac
        SELECTED+=("${MODELS[$i]}:${do_t}:${do_e}")
    done
else
    # Nessun filtro: tutti i modelli con flag globali
    for entry in "${MODELS[@]}"; do
        SELECTED+=("${entry}:${GLOBAL_TRAIN}:${GLOBAL_EVAL}")
    done
fi

# ── Costruisci la catena ──────────────────────────────────────────────────────
# Ogni riga: TYPE:CONFIG:TAG
> "$CHAIN_FILE"  # svuota/crea il file

MODEL_COUNT=0
for sel in "${SELECTED[@]}"; do
    # sel = "TAG:CONFIG:do_train:do_eval"
    TAG=$(echo "$sel" | cut -d: -f1)
    CFG=$(echo "$sel" | cut -d: -f2)
    DO_TRAIN=$(echo "$sel" | cut -d: -f3)
    DO_EVAL=$(echo "$sel" | cut -d: -f4)

    if [ "$DO_TRAIN" -eq 1 ]; then
        echo "train:${CFG}:${TAG}" >> "$CHAIN_FILE"
    fi
    if [ "$DO_EVAL" -eq 1 ]; then
        echo "eval:${CFG}:${TAG}" >> "$CHAIN_FILE"
    fi
    MODEL_COUNT=$((MODEL_COUNT + 1))
done

TOTAL=$(wc -l < "$CHAIN_FILE")

echo "============================================"
echo "  Multi-model GRPO Pipeline (self-chaining)"
echo "  Date:  $(date)"
echo "  Models: $MODEL_COUNT"
echo "  Total jobs: $TOTAL"
echo "============================================"
echo ""
echo "Catena:"
cat -n "$CHAIN_FILE"
echo ""

# ── Avvia il watcher in background (nohup) ────────────────────────────────────
# Il watcher controlla ogni 60s se la coda è vuota e sottomette il prossimo job.
mkdir -p logs

# Uccidi eventuale watcher precedente
if [ -f .chain_pid ]; then
    OLD_PID=$(cat .chain_pid)
    kill "$OLD_PID" 2>/dev/null && echo "Watcher precedente (PID $OLD_PID) terminato."
    rm -f .chain_pid
fi

nohup bash cluster/chain_next.sh >> logs/chain_watcher.log 2>&1 &
WATCHER_PID=$!
echo "$WATCHER_PID" > .chain_pid

echo ""
echo "============================================"
echo "  Pipeline avviata!"
echo "  Watcher PID: $WATCHER_PID"
echo "  Log: logs/chain_watcher.log"
echo "  Catena: .job_chain"
echo ""
echo "  Per monitorare:"
echo "    tail -f logs/chain_watcher.log"
echo "    myjobs"
echo ""
echo "  Per interrompere:"
echo "    kill \$(cat .chain_pid)   # uccidi il watcher"
echo "    killalljobs              # cancella il job SLURM attivo"
echo "============================================"
