# RunLogger

The official Python client for [Runlog](https://runlog.in/landing) — real-time ML training monitor.

Works with any training framework: PyTorch, JAX, TensorFlow, or plain Python.

---

## Table of Contents

1. [Installation](#installation)
2. [Quickstart](#quickstart)
3. [Initialization](#initialization)
4. [Logging Metrics](#logging-metrics)
5. [Logging Evaluations](#logging-evaluations)
6. [Runtime Knobs](#runtime-knobs)
7. [Pausing & Resuming](#pausing--resuming)
8. [Finishing a Run](#finishing-a-run)
9. [Logging Artifacts](#logging-artifacts)
10. [Offline Mode](#offline-mode)
11. [Manual Sync CLI](#manual-sync-cli)
12. [Auto Run Names](#auto-run-names)
13. [System Stats](#system-stats)
14. [Context Manager](#context-manager)
15. [Run Configuration](#run-configuration)
16. [Tags & Notes](#tags--notes)
17. [API Token](#api-token)
18. [Plans & Limits](#plans--limits)
19. [Error Handling](#error-handling)
20. [Full Example](#full-example)
21. [License](#license)
22. [FAQ](#faq)

---

## Installation

```bash
pip install runlog-sdk
```

**Or install from source:**

```bash
pip install git+https://github.com/runlog-in/runlog-sdk.git
```

## Quickstart

```python
import time
import math
import random
from runlogger import RunLogger

logger = RunLogger(
    base_url="https://runlog.in",
    project_name="my-project",
    api_token="rl-...",
)

for step in range(1000):
    # simulated loss — replace with your real training step
    loss = 3.5 * math.exp(-0.002 * step) + random.uniform(-0.05, 0.05)

    logger.log(step=step, loss=round(loss, 4))
    time.sleep(1)   # remove in real training

logger.finish()
```

---

## Initialization

```python
logger = RunLogger(
    base_url="https://runlog.in",
    api_token="rl-...",
    project_name="my-project",
    # run_name auto-generated if omitted
    config={...},                    # optional
    start_step=0,                    # optional
    metrics=[],                      # optional
    tags=[],                         # optional
    notes="",                        # optional
    log_system_stats=True,           # optional
    offline_mode=True,               # optional
    capture_terminal=True,           # optional
    verbose=False,                   # optional
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_url` | `str` | required | Your Runlog server URL. |
| `api_token` | `str` | required | Your personal API token from the dashboard. |
| `project_name` | `str` | required | The project this run belongs to. |
| `run_name` | `str` | auto-generated | A name for this run. Auto-generated if not provided. |
| `config` | `dict` | `{}` | Hyperparameters and settings for this run. |
| `start_step` | `int` | `0` | Step number this run starts from. Useful when resuming. |
| `metrics` | `list` | `[]` | Metric names you plan to log. Optional — metrics are tracked automatically. |
| `tags` | `list` | `[]` | Tags for grouping and filtering runs on the dashboard. |
| `notes` | `str` | `""` | Free-text notes about this run. |
| `log_system_stats` | `bool` | `True` | Automatically include GPU and CPU/RAM stats with every log call. |
| `offline_mode` | `bool` | `True` | Preserve data locally when connection is unavailable. See [Offline Mode](#offline-mode). |
| `capture_terminal` | `bool` | `True` | Capture stdout/stderr and stream terminal output to the dashboard alongside metrics. |
| `verbose` | `bool` | `False` | Print internal debug info — packet counts, intervals, orphan detail. Useful for troubleshooting. |

---


## Logging Metrics

```python
logger.log(step=step, loss=0.42, lr=0.001, grad_norm=1.2)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `step` | `int` | required — the current training step |
| `**kwargs` | `float / int / bool` | Any metrics you want to track |

### Returns

`True` if the log was accepted, `False` if logging has stopped.

### Notes

- Pass any number of metrics in a single call.
- Metric names are flexible — use whatever makes sense for your experiment.
- If `log_system_stats=True`, GPU and CPU stats are added automatically.
- If a daily log limit applies to your plan, RunLogger warns you when it is reached.

### Example

```python
import time
import math
import random

for step in range(1000):
    loss      = 3.5 * math.exp(-0.002 * step) + random.uniform(-0.05, 0.05)
    lr        = 3e-4 * min(1.0, step / 100)
    grad_norm = random.uniform(0.5, 1.5)

    logger.log(
        step=step,
        train_loss=round(loss, 4),
        lr=round(lr, 7),
        grad_norm=round(grad_norm, 4),
        tokens_per_sec=random.randint(58000, 66000),
    )
    time.sleep(1)   # remove in real training
```

---

## Logging Evaluations

```python
logger.log_eval(step=step, val_loss=0.38, perplexity=46.2)
```

Use `log_eval()` for metrics computed on a validation or test set. These are tracked separately from training metrics on the dashboard.

### Parameters

Same as `log()` — `step` is required, any additional metrics as keyword arguments.

### Returns

`True` if accepted, `False` otherwise.

### Example

```python
import time
import math
import random

for step in range(1000):
    loss = 3.5 * math.exp(-0.002 * step) + random.uniform(-0.05, 0.05)
    logger.log(step=step, loss=round(loss, 4))

    if step % 100 == 0 and step > 0:
        val_loss = loss + random.uniform(0.01, 0.05)   # simulated eval
        logger.log_eval(
            step=step,
            val_loss=round(val_loss, 4),
            perplexity=round(math.exp(val_loss), 2),
        )

    time.sleep(1)   # remove in real training
```

---

## Runtime Knobs

Runtime Knobs let you expose Python variables as live controls on the RunLogger dashboard. During training, collaborators with permission can adjust these values without restarting the process. Your script simply reads the latest value from `logger.knobs`.

Typical use cases include:

- Learning rate
- Batch size (apply at a safe synchronization point, such as the next epoch)
- Evaluation frequency
- Dropout
- Data augmentation strength
- Debug delays
- Custom application parameters

---

## Registering a Knob

```python
logger.register_knob(
    key="lr",
    value=1e-3,
    min=1e-5,
    max=1e-2,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key` | `str` | required | Unique identifier for the knob. |
| `value` | `float \| int` | required | Initial value shown on the dashboard. |
| `min` | `float \| int` | `0.0` | Minimum allowed value. |
| `max` | `float \| int` | `1.0` | Maximum allowed value. |
| `label` | `str` | `key` | Optional display name shown in the dashboard. |

A knob may be registered before training starts or dynamically while the run is already in progress.

---

## Reading Knob Values

Current values are available through the read-only `logger.knobs` dictionary.

```python
lr = logger.knobs.get("lr", 1e-3)
```

Read knob values wherever they affect your training loop. Values are synchronized automatically while the run is active.

---

## Example: Live Learning Rate

```python
logger.register_knob(
    "lr",
    value=1e-3,
    min=1e-5,
    max=1e-2,
)

for step in range(total_steps):

    lr = logger.knobs.get("lr", 1e-3)

    optimizer.param_groups[0]["lr"] = lr

    loss = train_step()

    logger.log(
        step=step,
        loss=loss,
        lr=lr,
    )
```

Move the **LR** dial in the dashboard and the training process begins using the updated value immediately.

---

## Example: Batch Size

Some parameters cannot safely change during an active optimization step. Batch size is typically applied between epochs.

```python
logger.register_knob(
    "batch_size",
    value=64,
    min=8,
    max=512,
)

current_batch_size = 64

for epoch in range(num_epochs):

    new_batch_size = int(
        logger.knobs.get("batch_size", current_batch_size)
    )

    if new_batch_size != current_batch_size:

        current_batch_size = new_batch_size

        train_loader = DataLoader(
            dataset,
            batch_size=current_batch_size,
            shuffle=True,
        )

    train_one_epoch(train_loader)
```

---

## Permissions

Runtime Knobs are visible to everyone who can view the run.

Only the project owner and collaborators with **Member** (or higher) permissions can modify knob values. Changes are synchronized to every connected client in real time.

---

## Offline Mode

When `offline_mode=True`, the latest knob values are preserved locally. If the training process temporarily disconnects or reconnects, the most recent values are restored automatically.

---

## Best Practices

- Read knob values immediately before using them.
- Apply structural changes such as batch size only at safe synchronization points (for example, between epochs).
- Use knobs to tune long-running experiments without restarting training.
- Register knobs at any point during a run to expose new runtime controls.
- Every knob change is recorded in the run timeline for reproducibility and auditing.

---

## Pausing & Resuming

You can pause a running training job from the Runlog dashboard. To support this, check `should_pause()` in your training loop:

```python
import time
import math
import random

for step in range(1000):
    loss = 3.5 * math.exp(-0.002 * step) + random.uniform(-0.05, 0.05)
    logger.log(step=step, loss=round(loss, 4))
    time.sleep(1)   # remove in real training

    if logger.should_pause():
        print("Training paused from dashboard")
        logger.finish(status="paused")
        break
```

### Returns

`True` if a pause has been requested from the dashboard, `False` otherwise.

### Notes

- Your training loop is responsible for stopping — RunLogger does not interrupt your code.
- When you click Resume on the dashboard, the pause flag is cleared automatically.

---

## Finishing a Run

Always call `finish()` at the end of your script:

```python
logger.finish()
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `status` | `str` | `"completed"` | Final status shown on the dashboard |

### Status values

| Status | When to use |
|---|---|
| `completed` | Training finished successfully |
| `crashed` | An error occurred |
| `paused` | Training was intentionally paused |

### Notes

- Always call `finish()` or use the [Context Manager](#context-manager). Runs left without a final status stay marked as `running` on the dashboard indefinitely.

---

## Logging Artifacts

```python
logger.log_artifact(
    path="checkpoints/step-1000.pt",
    name="checkpoint-step-1000",
    type="model",
    metadata={"step": 1000, "val_loss": 0.38},
)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` | Local file path to upload |
| `name` | `str` | Display name on the dashboard |
| `type` | `str` | Category — e.g. `"model"`, `"dataset"`, `"config"` |
| `metadata` | `dict` | Optional key-value info attached to the artifact |

### Returns

`True` if uploaded successfully, `False` if an error occurred.

### Notes

- Artifact upload blocks until complete.
- Large files may take time depending on your connection speed.

---

## Offline Mode

```python
logger = RunLogger(
    ...,
    offline_mode=True,   # default
)
```

When `offline_mode=True`, your data is preserved locally in a SQLite DB if the connection is unavailable. Once connectivity is restored, everything syncs to the dashboard automatically — in order, with no gaps.

When `offline_mode=False`, any data logged during a connection gap is lost.

### When to use each

| Scenario | Recommended |
|---|---|
| Long training runs | `offline_mode=True` |
| Unreliable or intermittent network | `offline_mode=True` |
| Short scripts, stable connection | Either |
| No local disk writes allowed | `offline_mode=False` |

### Plan requirement

Offline mode requires your Runlog plan to support it. If your plan does not include it, offline mode is disabled automatically at startup with a warning. If your plan is upgraded, it activates without restarting.

### Previous run recovery

If a previous run was interrupted while offline, its data is automatically recovered and synced the next time you start a run from the same directory. No manual steps required.

---

## Manual Sync CLI

If a run was interrupted and you want to sync its data manually without starting a new run, use the `runlogger-sync` command:

```bash
# scan the default dumps/ directory
runlogger-sync

# scan a specific directory
runlogger-sync --dir /path/to/runs

# sync one specific file
runlogger-sync --file dumps/.runlog_abc123.db

# show full debug output
runlogger-sync --verbose
runlogger-sync -v
```

### Options

| Flag | Description |
|---|---|
| `--dir` | Directory to scan (default: `dumps/`) |
| `--file` | Sync a single specific DB file |
| `--base-url` | Server URL — fallback if not in DB (or set `RUNLOGGER_URL`) |
| `--token` | API token — fallback if not in DB (or set `RUNLOGGER_TOKEN`) |
| `--verbose`, `-v` | Show debug detail: IDs, per-batch info, log uploads |

### Environment variables

```bash
export RUNLOGGER_URL=https://runlog.in
export RUNLOGGER_TOKEN=rl-...
runlogger-sync
```

### Notes

- The token and server URL are stored inside each DB file, so you usually don't need to pass them.
- Unrecoverable DB files (no token, no payload) are discarded automatically.
- Safe to run multiple times — already-synced packets are skipped.

---

## Auto Run Names

If you do not provide a `run_name`, one is generated automatically:

```
cosmic-nebula-42
silver-ridge-317
eager-summit-5
```

Format: `adjective-noun-number` — readable, memorable, and unique across any practical project scale.

---

## System Stats

If `pynvml` is installed, these GPU stats are included automatically with every `log()` call:

| Key | Description |
|---|---|
| `gpu_util` | GPU utilization % |
| `gpu_mem_used` | GPU memory used (MB) |
| `gpu_mem_total` | GPU total memory (MB) |

If `psutil` is installed, these CPU/RAM stats are included:

| Key | Description |
|---|---|
| `cpu_util` | CPU utilization % |
| `ram_used` | RAM used (MB) |
| `ram_total` | Total RAM (MB) |

To disable:

```python
logger = RunLogger(..., log_system_stats=False)
```

---

## Context Manager

```python
import time
import math
import random

with RunLogger(
    base_url="https://runlog.in",
    api_token="rl-...",
    project_name="my-project",
) as logger:
    for step in range(1000):
        loss = 3.5 * math.exp(-0.002 * step) + random.uniform(-0.05, 0.05)
        logger.log(step=step, loss=round(loss, 4))
        time.sleep(1)   # remove in real training
```

- Normal exit → `finish(status="completed")`
- Exception raised → `finish(status="crashed")`

The exception is not suppressed — it propagates normally after the run is closed.

---

## Run Configuration

```python
logger = RunLogger(
    ...,
    config={
        "model":        "68M",
        "architecture": "llama",
        "dataset":      "fineweb-10BT",
        "batch_size":   8,
        "seq_len":      1024,
        "lr":           3e-4,
        "warmup_steps": 100,
        "max_steps":    5000,
        "optimizer":    "adamw",
        "precision":    "bf16",
    },
)
```

Config is displayed on the run detail page. It is set once at initialization — pass all relevant hyperparameters upfront.

---

## Tags & Notes

### Tags

```python
logger = RunLogger(
    ...,
    tags=["baseline", "bf16", "fineweb"],
)
```

Tags appear on the dashboard and can be used to filter and group runs across a project.

### Notes

```python
logger = RunLogger(
    ...,
    notes="Testing SwiGLU vs GELU — same LR schedule, different FFN.",
)
```

Free-text notes visible on the run detail page.

---

## API Token

Get your API token from the Runlog dashboard under **Settings → API Tokens**.

Tokens start with `rl-`. Keep your token private — do not commit it to source control. Use environment variables:

```python
import os
from runlogger import RunLogger

logger = RunLogger(
    base_url="https://runlog.in",
    api_token=os.environ["RUNLOG_API_TOKEN"],
    project_name="my-project",
)
```

If an invalid token is provided, a `RuntimeError` is raised immediately at startup.

---

## Plans & Limits

Limits are applied automatically based on your plan — no configuration needed.

| Limit | What happens |
|---|---|
| **Daily log limit** | RunLogger warns you and stops logging for the rest of the day. Resets at midnight. |
| **Max metrics tracked** | Metric keys beyond your plan's limit are ignored. |
| **Log rate** | Data is recorded at the rate your plan allows. The most recent value always gets through. |
| **Offline mode** | Only available on supported plans. |

If your plan changes, limits update automatically — no restart needed.

---

## Error Handling

### Errors raised at startup

```python
RuntimeError: Invalid API token: ...
RuntimeError: [Runlog] account is banned.
```

### Everything else degrades gracefully

- Connection issues → data is preserved if offline mode is on, retried automatically
- Upload failures → logged to console, training continues unaffected

### Recommended pattern

```python
try:
    with RunLogger(...) as logger:
        for step in range(max_steps):
            loss = train()
            logger.log(step=step, loss=loss)
except RuntimeError as e:
    print(f"RunLogger error: {e}")
    # continue training without logging, or exit
```

---

## Full Example

```python
import math
import os
import random
import time
from runlogger import RunLogger
import time

# ── simulated training helpers ────────────────────────────────
# replace these with your real training code
_step_loss = [3.5]

def train_one_step():
    _step_loss[0] = _step_loss[0] * 0.9995 + random.uniform(-0.02, 0.02)
    time.sleep(0.05)   # remove in real training
    return max(0.1, _step_loss[0])

def evaluate():
    return _step_loss[0] + random.uniform(0.01, 0.05)

def get_lr(step, warmup=100, base_lr=3e-4):
    if step < warmup:
        return base_lr * step / warmup
    return base_lr * 0.5 * (1 + math.cos(math.pi * step / 5000))

def get_grad_norm():
    return random.uniform(0.5, 1.5)

def save_checkpoint(step):
    pass   # your checkpoint logic here
# ─────────────────────────────────────────────────────────────

logger = RunLogger(
    base_url="https://runlog.in",
    api_token=os.environ["RUNLOG_API_TOKEN"],
    project_name="koyna-v2",
    run_name="sft-run-1",
    offline_mode=True,
    config={
        "model":        "68M",
        "architecture": "llama",
        "dataset":      "mes_v2",
        "batch_size":   8,
        "seq_len":      1024,
        "lr":           3e-4,
        "warmup_steps": 100,
        "max_steps":    5000,
        "optimizer":    "adamw",
        "precision":    "bf16",
    },
    tags=["sft", "marathi", "68M"],
    notes="SFT on mes_v2 combined dataset, 883K samples.",
)

TOTAL_STEPS = 5000
EVAL_EVERY  = 500
best_val    = float("inf")

try:
    for step in range(1, TOTAL_STEPS + 1):
        loss      = train_one_step()
        lr        = get_lr(step)
        grad_norm = get_grad_norm()

        logger.log(
            step=step,
            loss=round(loss, 4),
            lr=round(lr, 7),
            grad_norm=round(grad_norm, 4),
        )

        if step % EVAL_EVERY == 0:
            val_loss = evaluate()
            is_best  = val_loss < best_val
            if is_best:
                best_val = val_loss
                save_checkpoint(step)
                logger.log_artifact(
                    path=f"checkpoints/step-{step}.pt",
                    name=f"checkpoint-step-{step}",
                    type="model",
                    metadata={"step": step, "val_loss": val_loss},
                )
            logger.log_eval(
                step=step,
                val_loss=round(val_loss, 4),
                perplexity=round(math.exp(val_loss), 2),
                is_best=is_best,
            )

        if logger.should_pause():
            print("Paused from dashboard")
            logger.finish(status="paused")
            break
        time.sleep(1)
    else:
        logger.finish(status="completed")

except KeyboardInterrupt:
    logger.finish(status="crashed")

except Exception as e:
    logger.finish(status="crashed")
    raise
```

---

## License

RunLogger is distributed under the **Business Source License 1.1**.

- Free to use for any purpose when connecting to [runlog.in](https://runlog.in)
- Modification permitted for personal use only
- Redistribution or use with other servers requires written permission

For commercial licensing: [runlog.uk@gmail.com](mailto:runlog.uk@gmail.com)

Full terms: [runlog.in/auth/terms](https://runlog.in/auth/terms)

---

## FAQ

**Do I need to call `finish()` if I use the context manager?**
No — it is called automatically.

**What if I forget `finish()`?**
The run stays marked as `running` on the dashboard. Always call `finish()` or use the context manager.

**Can I use RunLogger with multi-GPU / DDP training?**
Yes. Log only from rank 0 to avoid duplicate data:
```python
if rank == 0:
    logger.log(step=step, loss=loss)
```

**Can I log string values as metrics?**
No — metric values must be `int`, `float`, or `bool`. Pass strings in `config`, `tags`, or `notes` instead.

**Can I have multiple loggers in one script?**
Yes. Each `RunLogger` instance is independent and creates its own run.

**Does RunLogger affect training performance?**
No. All logging is non-blocking — your training loop is not slowed down.

**What if my machine is killed mid-training?**
If `offline_mode=True`, all data logged before the crash is preserved and recovered automatically the next time you start a run from the same directory.

**Can I use RunLogger with a self-hosted Runlog instance?**
Yes. Set `base_url` to your own server address.

**How do I debug connection or sync issues?**
Pass `verbose=True` to `RunLogger(...)` or use `runlogger-sync --verbose` to see full internal detail.

**Where are offline DB files stored?**
In a `dumps/` directory relative to where your training script runs. Files are named `.runlog_<run_id>.db` and cleaned up automatically after sync.
