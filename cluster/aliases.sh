#!/bin/bash
# ============================================================================
# Alias utili per il cluster DMI
#
# Uso:
#   source cluster/aliases.sh
#
# Per caricarli automaticamente, aggiungi al tuo ~/.bashrc:
#   source ~/GRPO-strict-generation/cluster/aliases.sh
# ============================================================================

PROJ_DIR="$HOME/GRPO-strict-generation"

# ── Job management ───────────────────────────────────────────────────────────

# Controlla i miei job attivi
alias myjobs='squeue --me --format="%.10i %.20j %.8T %.10M %.6D %.20R %o"'

# Info dettagliata su un job (uso: jobinfo <JOB_ID>)
jobinfo() {
    if [ -z "$1" ]; then
        echo "Uso: jobinfo <JOB_ID>"
        return 1
    fi
    scontrol show job "$1"
}

# Cancella un job (uso: killjob <JOB_ID>)
alias killjob='scancel'

# Cancella tutti i miei job
alias killalljobs='scancel --me'

# ── Log monitoring ───────────────────────────────────────────────────────────

# Segui il log di un job di training GRPO (uso: grpolog <JOB_ID>)
grpolog() {
    if [ -z "$1" ]; then
        echo "Uso: grpolog <JOB_ID>"
        return 1
    fi
    local logfile="$PROJ_DIR/logs/slurm-grpo-${1}.log"
    if [ ! -f "$logfile" ]; then
        echo "Log non trovato: $logfile"
        return 1
    fi
    tail -f "$logfile"
}

# Segui il log di un job di eval GRPO (uso: evallog <JOB_ID>)
evallog() {
    if [ -z "$1" ]; then
        echo "Uso: evallog <JOB_ID>"
        return 1
    fi
    local logfile="$PROJ_DIR/logs/slurm-eval-${1}.log"
    if [ ! -f "$logfile" ]; then
        echo "Log non trovato: $logfile"
        return 1
    fi
    tail -f "$logfile"
}

# Segui il log di un job baseline (uso: baselog <JOB_ID>)
baselog() {
    if [ -z "$1" ]; then
        echo "Uso: baselog <JOB_ID>"
        return 1
    fi
    local logfile="$PROJ_DIR/logs/slurm-baseline-${1}.log"
    if [ ! -f "$logfile" ]; then
        echo "Log non trovato: $logfile"
        return 1
    fi
    tail -f "$logfile"
}

# Mostra l'ultimo log (qualsiasi tipo) — uso: lastlog
lastlog() {
    local logfile
    logfile=$(ls -t "$PROJ_DIR"/logs/slurm*.log 2>/dev/null | head -1)
    if [ -z "$logfile" ]; then
        echo "Nessun log trovato in $PROJ_DIR/logs/"
        return 1
    fi
    echo "==> $logfile <=="
    tail -f "$logfile"
}

# ── Filesystem ───────────────────────────────────────────────────────────────

# Tree ricorsivo di una cartella (uso: tree <DIR> [DEPTH])
tree() {
    local dir="${1:-.}"
    local depth="${2:-3}"
    find "$dir" -maxdepth "$depth" | sed -e "s|[^/]*/|  |g" -e "s|  |├─|"
}

# Alias più leggibile basato su ls -R (uso: ltree <DIR>)
ltree() {
    local dir="${1:-.}"
    ls -R "$dir" | grep ":$" | sed -e 's/:$//' -e 's/[^-][^\/]*\//──/g' -e 's/^/  /' -e 's/─/│/'
}

# ── GPU & risorse ────────────────────────────────────────────────────────────

# Stato GPU — funziona solo dentro un job SLURM con GPU allocata
alias gpu='nvidia-smi'

# Uso disco del progetto
alias quota='quota -s'

# ── Quick commands ───────────────────────────────────────────────────────────

# Vai alla directory del progetto
alias proj='cd "$PROJ_DIR"'

# Mostra i checkpoint disponibili
ckpts() {
    echo "=== Checkpoint intermedi (per stage) ==="
    for d in "$PROJ_DIR"/experiments/checkpoints/grpo/stage_*/; do
        if [ -d "$d" ]; then
            echo "  $(basename "$d"):"
            ls -d "$d"checkpoint-* 2>/dev/null | while read -r c; do echo "    $(basename "$c")"; done
        fi
    done
    echo ""
    echo "=== Modelli finali (stages/) ==="
    ls -d "$PROJ_DIR"/experiments/checkpoints/grpo/stages/stage_*/ 2>/dev/null | while read -r s; do
        echo "  $(basename "$s")"
    done || echo "  (nessuno)"
}

# Lancia training
alias train='cd "$PROJ_DIR" && sbatch cluster/train.sh'

# Lancia eval
alias eval-grpo='cd "$PROJ_DIR" && sbatch cluster/eval_grpo.sh'
alias eval-curriculum='cd "$PROJ_DIR" && CURRICULUM=1 sbatch cluster/eval_grpo.sh'
alias eval-baseline='cd "$PROJ_DIR" && sbatch cluster/eval_baseline.sh'

# ── Meta ─────────────────────────────────────────────────────────────────────

# Lista di tutti i comandi custom registrati
_GRPO_ALIASES="myjobs jobinfo killjob killalljobs grpolog evallog baselog lastlog tree ltree gpu quota proj ckpts train eval-grpo eval-curriculum eval-baseline claudio unload-aliases"

# Mostra i comandi disponibili
claudio() {
    echo "Comandi GRPO disponibili:"
    echo "   myjobs            — lista job attivi"
    echo "   jobinfo <ID>      — dettagli job"
    echo "   killjob <ID>      — cancella job"
    echo "   killalljobs       — cancella tutti i miei job"
    echo "   grpolog <ID>      — segui log training GRPO"
    echo "   evallog <ID>      — segui log eval GRPO"
    echo "   baselog <ID>      — segui log baseline"
    echo "   lastlog           — segui l'ultimo log"
    echo "   tree <DIR> [N]    — albero cartelle (profondità N)"
    echo "   ltree <DIR>       — albero cartelle compatto"
    echo "   gpu               — stato GPU"
    echo "   quota             — uso disco progetto"
    echo "   proj              — cd al progetto"
    echo "   ckpts             — mostra checkpoint"
    echo "   train             — sbatch training"
    echo "   eval-grpo         — sbatch eval GRPO"
    echo "   eval-curriculum   — sbatch eval curriculum"
    echo "   eval-baseline     — sbatch eval baseline"
    echo "   claudio             — mostra questo messaggio"
    echo "   unload-aliases    — rimuovi tutti i comandi custom"
}

# Rimuovi tutti gli alias e funzioni custom
unload-aliases() {
    for cmd in $_GRPO_ALIASES; do
        unalias "$cmd" 2>/dev/null
        unset -f "$cmd" 2>/dev/null
    done
    unset _GRPO_ALIASES PROJ_DIR
    echo "✅ Alias GRPO rimossi."
}

echo "✅ Alias GRPO caricati. Digita 'claudio' per la lista comandi."
