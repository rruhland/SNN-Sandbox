$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $python = $pyLauncher.Source
    } else {
        $python = "python"
    }
}

if ((Split-Path -Leaf $python) -eq "py.exe") {
    & $python -3 -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
} else {
    & $python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
}
