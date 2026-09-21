param(
  [string]$Root = (Split-Path -Parent $MyInvocation.MyCommand.Path),
  [string]$UvExe = '',
  [string]$PythonExe = '',
  [switch]$SkipCudaProbe
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
Set-Location -LiteralPath $Root

$FlashAttentionVersion = '2.8.3'
$WheelUrl = 'https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu128torch2.8.0cxx11abiFALSE-cp311-cp311-win_amd64.whl'
$WheelSha256 = 'AF74177E0D63E45D154963031944E12B7E935F92EA6FC13AE9203B5C44464F06'
$Venv = Join-Path $Root '.venv'
$Downloads = Join-Path $Root '.runtime\downloads'
$WheelPath = Join-Path $Downloads 'flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp311-cp311-win_amd64.whl'
if (-not $UvExe) { $UvExe = Join-Path $Root '.runtime\uv\uv.exe' }
if (-not $PythonExe) { $PythonExe = Join-Path $Venv 'Scripts\python.exe' }

function Info([string]$Text) { Write-Host "[INFO] $Text" -ForegroundColor Cyan }
function Ok([string]$Text) { Write-Host "[OK] $Text" -ForegroundColor Green }
function Fail([string]$Text) { throw $Text }

if (-not (Test-Path -LiteralPath $UvExe)) { Fail "uv was not found at $UvExe" }
if (-not (Test-Path -LiteralPath $PythonExe)) { Fail "The project Python was not found at $PythonExe" }
if (-not (Test-Path -LiteralPath (Join-Path $Root 'uv.lock'))) { Fail 'uv.lock is required before installing FlashAttention.' }
New-Item -ItemType Directory -Force -Path $Downloads | Out-Null

$lockText = Get-Content -LiteralPath (Join-Path $Root 'uv.lock') -Raw
if ($lockText -notmatch [regex]::Escape($WheelUrl) -or $lockText -notmatch $WheelSha256.ToLowerInvariant()) {
  Fail 'uv.lock does not contain the expected FlashAttention wheel URL and SHA-256 pin.'
}

Info "Installing the pinned Windows FlashAttention wheel for Torch 2.8.0/cu128."
Info "Wheel: $WheelUrl"
Info "SHA-256: $WheelSha256"

if (Test-Path -LiteralPath $WheelPath) {
  $existingHash = (Get-FileHash -LiteralPath $WheelPath -Algorithm SHA256).Hash
  if ($existingHash -ne $WheelSha256) {
    Remove-Item -LiteralPath $WheelPath -Force
  }
}
if (-not (Test-Path -LiteralPath $WheelPath)) {
  $downloaded = $false
  try {
    & curl.exe --fail --location --retry 4 --retry-delay 2 --output $WheelPath $WheelUrl
    $downloaded = ($LASTEXITCODE -eq 0)
  } catch {
    $downloaded = $false
  }
  if (-not $downloaded) {
    Remove-Item -LiteralPath $WheelPath -Force -ErrorAction SilentlyContinue
    try {
      Invoke-WebRequest -Uri $WheelUrl -OutFile $WheelPath -UseBasicParsing
      $downloaded = $true
    } catch {
      $downloaded = $false
    }
  }
  if (-not $downloaded) { Fail 'The pinned FlashAttention wheel could not be downloaded.' }
}
$actualHash = (Get-FileHash -LiteralPath $WheelPath -Algorithm SHA256).Hash
if ($actualHash -ne $WheelSha256) {
  Remove-Item -LiteralPath $WheelPath -Force
  Fail "FlashAttention wheel hash mismatch. Expected $WheelSha256, got $actualHash."
}

& $UvExe pip install --python $PythonExe --no-cache --no-deps $WheelPath
if ($LASTEXITCODE -ne 0) { Fail 'uv failed to install the locked FlashAttention Windows wheel.' }

$finalProbe = @"
from importlib.metadata import version
import torch
import flash_attn
import flash_attn_2_cuda
assert version('flash-attn') == '$FlashAttentionVersion', version('flash-attn')
assert torch.__version__.split('+')[0] == '2.8.0', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
print('FlashAttention import OK |', version('flash-attn'), '| Torch', torch.__version__, '| CUDA', torch.version.cuda)
"@
& $PythonExe -c $finalProbe
if ($LASTEXITCODE -ne 0) { Fail 'The pinned FlashAttention wheel failed the Torch/CUDA ABI import probe.' }

if (-not $SkipCudaProbe) {
  $cudaProbe = @"
import torch
from flash_attn import flash_attn_func
assert torch.cuda.is_available(), 'CUDA is not available for the FlashAttention forward probe'
device = torch.device('cuda')
q = torch.randn((1, 128, 8, 64), device=device, dtype=torch.float16)
k = torch.randn((1, 128, 8, 64), device=device, dtype=torch.float16)
v = torch.randn((1, 128, 8, 64), device=device, dtype=torch.float16)
out = flash_attn_func(q, k, v, causal=False)
torch.cuda.synchronize()
assert tuple(out.shape) == tuple(q.shape), (out.shape, q.shape)
print('FlashAttention CUDA forward OK |', tuple(out.shape), '|', torch.cuda.get_device_name(0))
"@
  & $PythonExe -c $cudaProbe
  if ($LASTEXITCODE -ne 0) { Fail 'The FlashAttention wheel imported, but its real CUDA forward probe failed.' }
}
Ok 'Pinned FlashAttention Windows wheel installed and validated.'
