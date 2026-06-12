param(
    [string]$HostName = "0.0.0.0",
    [int]$Port = 8000,
    [switch]$AllowCpu
)

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
    $argsList = @("-3", "-m", "snn_sandbox.server", "--host", $HostName, "--port", "$Port")
} else {
    $argsList = @("-m", "snn_sandbox.server", "--host", $HostName, "--port", "$Port")
}
if ($AllowCpu) {
    $argsList += "--allow-cpu"
}

& $python @argsList
