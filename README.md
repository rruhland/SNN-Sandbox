# SNN Sandbox

Desktop-hosted spiking neural network sandbox for local/STDP experiments. The first experiment is intentionally small: one-bit addition (`0+0`, `0+1`, `1+0`, `1+1`) with a browser visualizer that controls a live backend model.

## Architecture

- Desktop runs `snn_sandbox.server` and owns the live model state.
- PyTorch tensors run on CUDA by default. The code does not use autograd or backpropagation.
- Input to hidden learning is local STDP from spike traces.
- Hidden to output learning is reward-modulated STDP from hidden activity and correctness.
- The browser UI controls the trainer, but never runs training.
- The top trainer dashboard streams only cheap live metrics: running state, trials, trials/sec, accuracy, last result, and recent accuracy history.
- The bottom model inspector is frozen. It updates only when `Pull Current` exports a model snapshot.
- Pull exports weights, recent spikes, config, confusion, and stats for browser-side inspection while training continues.

## Quick Start

Create a Python environment on the desktop GPU machine, then install a CUDA-enabled PyTorch build that matches your driver.

```powershell
cd "C:\Users\rruhl\Documents\SNN Sandbox"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.\scripts\check_cuda.ps1
.\scripts\run_desktop_server.ps1
```

Open the UI on the desktop:

```text
http://127.0.0.1:8000
```

For another device on the same private network or Tailscale, start the server with the default `0.0.0.0` host and open:

```text
http://DESKTOP_NAME_OR_TAILSCALE_IP:8000
```

If you only need a slow smoke test on a machine without CUDA:

```powershell
.\scripts\run_desktop_server.ps1 -AllowCpu
```

## Controls

- `Start` and `Stop` toggle the desktop training loop.
- `Reset` creates a fresh network.
- `Speed` changes training episodes per backend tick.
- `Save` writes current weights and metrics into `states/`.
- `Load` restores a saved state.
- `Pull Current` exports a frozen model snapshot into the bottom inspector without stopping training.

## Developer Smoke Test

```powershell
.\.venv\Scripts\python.exe -m snn_sandbox.smoke
```

The bundled Codex Python runtime can also run the smoke test with `--allow-cpu` behavior, but production training should run on the desktop CUDA environment.
