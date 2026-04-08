#!/bin/bash
# ============================================================================
# Watcher — Esegue i job dalla catena .job_chain uno alla volta.
#
# Gira sul login node (NON dentro un job SLURM). Controlla ogni 60s
# se la coda è vuota e sottomette il prossimo job.
#
# Se un job SLURM fallisce (exit code != 0, CANCELLED), il watcher si
# blocca, scrive .chain_failed con le info del job, e NON sottomette
# altri job. Usare run_all.sh --resume per riprendere.
#
# Se un job di TRAINING va in TIMEOUT (tempo limite superato), il watcher
# lo rilancia automaticamente con --resume (max 2 tentativi). Se anche
# il retry va in timeout, la pipeline si ferma normalmente.
#
# Uso:
#   nohup bash cluster/chain_next.sh >> logs/chain_watcher.log 2>&1 &
#
# Per interrompere: kill $(cat .chain_pid), oppure cancella .job_chain
# ============================================================================

PROJ_DIR="$HOME/GRPO-strict-generation"
CHAIN_FILE="$PROJ_DIR/.job_chain"
FAILED_FILE="$PROJ_DIR/.chain_failed"
POLL_INTERVAL=60  # secondi tra un check e l'altro
MAX_TIMEOUT_RETRIES=2  # max auto-resume per TIMEOUT sullo stesso job

cd "$PROJ_DIR"

# Registra il PID così kill $(cat .chain_pid) funziona anche al riavvio manuale
echo $$ > "$PROJ_DIR/.chain_pid"

echo "[chain] Watcher avviato (PID $$) — $(date)"
echo "[chain] File catena: $CHAIN_FILE"

LAST_JOB_ID=""
LAST_JOB_DESC=""
TIMEOUT_RETRIES=0  # contatore retry per il job corrente
LAST_RETRY_TAG=""  # tag del job in retry (reset quando cambia job)

while true; do
    # Se non c'è più il file catena, abbiamo finito
    if [ ! -f "$CHAIN_FILE" ] || [ ! -s "$CHAIN_FILE" ]; then
        # Ma controlla che l'ultimo job non sia fallito
        if [ -n "$LAST_JOB_ID" ]; then
            sleep 5  # attendi che sacct aggiorni lo stato
            EXIT_CODE=$(sacct -j "$LAST_JOB_ID" --format=ExitCode --noheader --parsable2 2>/dev/null | head -1 | cut -d: -f1)
            STATE=$(sacct -j "$LAST_JOB_ID" --format=State --noheader --parsable2 2>/dev/null | head -1)

            FINAL_FAILED=0
            if [ "$STATE" = "TIMEOUT" ] || [ "$STATE" = "CANCELLED" ] || [ "$STATE" = "CANCELLED+" ]; then
                FINAL_FAILED=1
            elif [ -n "$EXIT_CODE" ] && [ "$EXIT_CODE" != "0" ]; then
                FINAL_FAILED=1
            fi

            if [ "$FINAL_FAILED" -eq 1 ]; then
                # Se era un train in TIMEOUT e abbiamo retry disponibili, ri-accoda
                FINAL_TYPE=$(echo "$LAST_JOB_DESC" | cut -d: -f1)
                FINAL_CFG=$(echo "$LAST_JOB_DESC" | cut -d: -f2)
                FINAL_TAG=$(echo "$LAST_JOB_DESC" | cut -d: -f3)

                if [ "$STATE" = "TIMEOUT" ] && [ "$FINAL_TYPE" = "train" ]; then
                    if [ "$FINAL_TAG" = "$LAST_RETRY_TAG" ]; then
                        TIMEOUT_RETRIES=$((TIMEOUT_RETRIES + 1))
                    else
                        TIMEOUT_RETRIES=1
                        LAST_RETRY_TAG="$FINAL_TAG"
                    fi
                    if [ "$TIMEOUT_RETRIES" -le "$MAX_TIMEOUT_RETRIES" ]; then
                        echo "[chain] ⏰ Ultimo job $LAST_JOB_ID ($FINAL_TAG) TIMEOUT — auto-resume ($TIMEOUT_RETRIES/$MAX_TIMEOUT_RETRIES) — $(date)"
                        echo "train:${FINAL_CFG}:${FINAL_TAG}:--resume" > "$CHAIN_FILE"
                        LAST_JOB_ID=""
                        sleep 5
                        continue
                    fi
                fi

                echo "[chain] ❌ Ultimo job $LAST_JOB_ID ($LAST_JOB_DESC) FALLITO (state=$STATE exit=$EXIT_CODE) — $(date)"
                echo "${LAST_JOB_DESC}" > "$FAILED_FILE"
                echo "[chain] Pipeline interrotta. Usa: bash cluster/run_all.sh --resume"
                rm -f "$PROJ_DIR/.chain_pid"
                exit 1
            fi
        fi
        echo "[chain] ✅ Pipeline completata! — $(date)"
        rm -f "$CHAIN_FILE" "$PROJ_DIR/.chain_pid" "$FAILED_FILE"
        exit 0
    fi

    # Controlla se ci sono job attivi (RUNNING o PENDING)
    ACTIVE=$(squeue --me --noheader 2>/dev/null | wc -l)
    if [ "$ACTIVE" -gt 0 ]; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    # ── Coda vuota: controlla se l'ultimo job è fallito ───────────────────
    if [ -n "$LAST_JOB_ID" ]; then
        sleep 5  # attendi aggiornamento sacct
        EXIT_CODE=$(sacct -j "$LAST_JOB_ID" --format=ExitCode --noheader --parsable2 2>/dev/null | head -1 | cut -d: -f1)
        STATE=$(sacct -j "$LAST_JOB_ID" --format=State --noheader --parsable2 2>/dev/null | head -1)

        # Determina se il job è fallito: exit code != 0 OPPURE stato anomalo
        JOB_FAILED=0
        IS_TIMEOUT=0
        if [ "$STATE" = "TIMEOUT" ]; then
            JOB_FAILED=1
            IS_TIMEOUT=1
        elif [ "$STATE" = "CANCELLED" ] || [ "$STATE" = "CANCELLED+" ]; then
            JOB_FAILED=1
        elif [ -n "$EXIT_CODE" ] && [ "$EXIT_CODE" != "0" ]; then
            JOB_FAILED=1
        fi

        if [ "$JOB_FAILED" -eq 1 ]; then
            FAILED_TYPE=$(echo "$LAST_JOB_DESC" | cut -d: -f1)
            FAILED_CFG=$(echo "$LAST_JOB_DESC" | cut -d: -f2)
            FAILED_TAG=$(echo "$LAST_JOB_DESC" | cut -d: -f3)

            # ── TIMEOUT su un train: auto-resume ──────────────────────
            if [ "$IS_TIMEOUT" -eq 1 ] && [ "$FAILED_TYPE" = "train" ]; then
                # Traccia retry per lo stesso tag
                if [ "$FAILED_TAG" = "$LAST_RETRY_TAG" ]; then
                    TIMEOUT_RETRIES=$((TIMEOUT_RETRIES + 1))
                else
                    TIMEOUT_RETRIES=1
                    LAST_RETRY_TAG="$FAILED_TAG"
                fi

                if [ "$TIMEOUT_RETRIES" -le "$MAX_TIMEOUT_RETRIES" ]; then
                    echo "[chain] ⏰ Job $LAST_JOB_ID ($FAILED_TAG) TIMEOUT — auto-resume ($TIMEOUT_RETRIES/$MAX_TIMEOUT_RETRIES) — $(date)"

                    # Reinserisci il train con --resume + l'eval in testa alla catena
                    RESUME_CHAIN=$(mktemp)
                    echo "train:${FAILED_CFG}:${FAILED_TAG}:--resume" > "$RESUME_CHAIN"
                    [ -f "$CHAIN_FILE" ] && [ -s "$CHAIN_FILE" ] && cat "$CHAIN_FILE" >> "$RESUME_CHAIN"
                    mv "$RESUME_CHAIN" "$CHAIN_FILE"

                    LAST_JOB_ID=""
                    sleep 5
                    continue
                else
                    echo "[chain] ❌ Job $LAST_JOB_ID ($FAILED_TAG) TIMEOUT — max retry ($MAX_TIMEOUT_RETRIES) raggiunto — $(date)"
                fi
            else
                echo "[chain] ❌ Job $LAST_JOB_ID ($LAST_JOB_DESC) FALLITO — state=$STATE exit=$EXIT_CODE — $(date)"
            fi

            # ── Fallimento definitivo: ferma la pipeline ──────────────
            if [ "$FAILED_TYPE" = "train" ] && [ -f "$CHAIN_FILE" ] && [ -s "$CHAIN_FILE" ]; then
                NEXT_IN_CHAIN=$(head -1 "$CHAIN_FILE")
                NEXT_TYPE=$(echo "$NEXT_IN_CHAIN" | cut -d: -f1)
                NEXT_TAG=$(echo "$NEXT_IN_CHAIN" | cut -d: -f3)
                if [ "$NEXT_TYPE" = "eval" ] && [ "$NEXT_TAG" = "$FAILED_TAG" ]; then
                    echo "[chain] ⏭  Rimosso eval di $FAILED_TAG dalla catena (train fallito)"
                    tail -n +2 "$CHAIN_FILE" > "$CHAIN_FILE.tmp" && mv "$CHAIN_FILE.tmp" "$CHAIN_FILE"
                    [ ! -s "$CHAIN_FILE" ] && rm -f "$CHAIN_FILE"
                fi
            fi

            echo "${LAST_JOB_DESC}" > "$FAILED_FILE"
            REMAINING=$([ -f "$CHAIN_FILE" ] && wc -l < "$CHAIN_FILE" || echo 0)
            echo "[chain] Pipeline interrotta. Rimanenti: $REMAINING job"
            echo "[chain] Per riprendere: bash cluster/run_all.sh --resume"
            rm -f "$PROJ_DIR/.chain_pid"
            exit 1
        fi
        echo "[chain] ✓ Job $LAST_JOB_ID ($LAST_JOB_DESC) completato — state=$STATE — $(date)"

        # Reset contatore retry quando un job completa con successo
        TIMEOUT_RETRIES=0
        LAST_RETRY_TAG=""
    fi

    # ── Sottometti il prossimo job ────────────────────────────────────────
    NEXT=$(head -1 "$CHAIN_FILE")

    # Rimuovi la prima riga
    tail -n +2 "$CHAIN_FILE" > "$CHAIN_FILE.tmp" && mv "$CHAIN_FILE.tmp" "$CHAIN_FILE"
    if [ ! -s "$CHAIN_FILE" ]; then
        rm -f "$CHAIN_FILE"
    fi

    # Parsing: TYPE:CONFIG:TAG[:EXTRA]
    TYPE=$(echo "$NEXT" | cut -d: -f1)
    CFG=$(echo "$NEXT" | cut -d: -f2)
    TAG=$(echo "$NEXT" | cut -d: -f3)
    EXTRA=$(echo "$NEXT" | cut -d: -f4-)

    # Guard: config vuoto → pipeline fallita
    if [ -z "$CFG" ]; then
        echo "[chain] ❌ Config vuoto per $TYPE $TAG — catena corrotta"
        echo "${NEXT}" > "$FAILED_FILE"
        echo "[chain] Pipeline interrotta. Per riprendere: bash cluster/run_all.sh --resume"
        rm -f "$PROJ_DIR/.chain_pid"
        exit 1
    fi

    REMAINING=$([ -f "$CHAIN_FILE" ] && wc -l < "$CHAIN_FILE" || echo 0)
    echo "[chain] Sottometto: $TYPE $TAG ($CFG) extra='$EXTRA' — $REMAINING rimanenti — $(date)"

    LAST_JOB_DESC="$NEXT"

    case "$TYPE" in
        train)
            LAST_JOB_ID=$(CONFIG="$CFG" EXTRA_ARGS="$EXTRA" sbatch --job-name="train-${TAG}" --parsable cluster/train.sh)
            ;;
        eval)
            # EXTRA can contain --skip-stages=N for resume
            SKIP_N=0
            if echo "$EXTRA" | grep -qP '^--skip-stages=\d+$'; then
                SKIP_N=$(echo "$EXTRA" | grep -oP '\d+')
            fi
            # Auto-detect curriculum from config YAML (don't hardcode CURRICULUM=1)
            IS_CURRICULUM=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('${CFG}'))
c = cfg.get('curriculum', {})
print('1' if c and c.get('enabled', False) else '0')
" 2>/dev/null || echo "0")
            LAST_JOB_ID=$(CONFIG="$CFG" COMPARE=1 CURRICULUM="$IS_CURRICULUM" SKIP_STAGES="$SKIP_N" sbatch --job-name="eval-${TAG}" --parsable cluster/eval.sh)
            ;;
        *)
            echo "[chain] ❌ Tipo sconosciuto: $TYPE — skip"
            LAST_JOB_ID=""
            continue
            ;;
    esac

    echo "[chain] Job ID: $LAST_JOB_ID"

    # Aspetta un po' prima di ricontrollare
    sleep 10
done
