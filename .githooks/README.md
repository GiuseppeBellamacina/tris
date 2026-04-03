# Git Hooks

Hook personalizzati per questa repo. Vengono eseguiti automaticamente da git.

## Hook disponibili

### `pre-push`

Ogni volta che fai `git push`, sincronizza automaticamente sul cluster DMI:

- `src/`
- `cluster/`
- `experiments/configs/`

Usa `sync_cluster.ps1 -Action push` per ogni cartella. Se il cluster non è raggiungibile o il sync fallisce, il push su GitHub prosegue comunque.

## Configurazione

Dopo aver clonato la repo su un nuovo PC, esegui:

```bash
git config core.hooksPath .githooks
```

Questo dice a git di usare `.githooks/` invece della cartella default `.git/hooks/`. L'impostazione è locale alla repo.

> **Nota**: su Linux/macOS potrebbe servire rendere eseguibile l'hook:
> ```bash
> chmod +x .githooks/pre-push
> ```

## Disabilitare temporaneamente

Per fare un push senza triggerare il sync:

```bash
git push --no-verify
```
