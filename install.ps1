$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# The runtime is project-local and the installer is idempotent. Existing
# models, datasets, samples, projects and a healthy .venv are preserved.
$UvVersion='0.11.33'
$PythonVersion='3.11.15'
$TorchVersion='2.8.0'
$FlashAttentionVersion='2.8.3'
$Runtime=Join-Path $Root '.runtime'
$UvDir=Join-Path $Runtime 'uv'
$UvExe=Join-Path $UvDir 'uv.exe'
$Venv=Join-Path $Root '.venv'
$PythonExe=Join-Path $Venv 'Scripts\python.exe'
$Downloads=Join-Path $Runtime 'downloads'
$Temp=Join-Path $Runtime 'temp'
$UvPython=Join-Path $Runtime 'python'
$HfHome=Join-Path $Runtime 'cache\huggingface'
foreach($d in @($Runtime,$UvDir,$Downloads,$Temp,$UvPython,$HfHome,(Join-Path $HfHome 'xet'),(Join-Path $Root 'models'),(Join-Path $Root 'outputs'),(Join-Path $Root 'samples'),(Join-Path $Root 'datasets'),(Join-Path $Root 'projects'),(Join-Path $Root 'training'),(Join-Path $Root 'base_models'),(Join-Path $Root 'loras'))){New-Item -ItemType Directory -Force -Path $d|Out-Null}
$env:UV_PYTHON_INSTALL_DIR=$UvPython;$env:UV_PYTHON_PREFERENCE='managed';$env:UV_NO_CACHE='1';$env:PIP_NO_CACHE_DIR='1';$env:UV_LINK_MODE='copy';$env:PYTHONUTF8='1';$env:PYTHONDONTWRITEBYTECODE='1';$env:TEMP=$Temp;$env:TMP=$Temp;$env:HF_HOME=$HfHome;$env:HF_XET_CACHE=Join-Path $HfHome 'xet'
function Section($t){Write-Host "`n=== $t ===" -ForegroundColor Cyan}
function Download($url,$dst){
  if(Test-Path $dst){return}
  $ok=$false
  try { & curl.exe --fail --location --retry 4 --retry-delay 2 --output $dst $url; $ok=($LASTEXITCODE -eq 0 -and (Test-Path $dst) -and ((Get-Item $dst).Length -gt 0)) } catch { $ok=$false }
  if(-not $ok){ Remove-Item -LiteralPath $dst -Force -ErrorAction SilentlyContinue; Invoke-WebRequest -Uri $url -OutFile $dst -UseBasicParsing }
  if(!(Test-Path $dst) -or (Get-Item $dst).Length -le 0){throw "Download failed: $url"}
}
function Run($file,[string[]]$Arguments){& $file @Arguments;if($LASTEXITCODE -ne 0){throw "Command failed ($LASTEXITCODE): $file $($Arguments -join ' ')"}}

Section '1/5 - Local uv'
if(!(Test-Path $UvExe)){
  $zip=Join-Path $Downloads "uv-$UvVersion.zip"
  Download "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" $zip
  tar.exe -xf $zip -C $UvDir
  if($LASTEXITCODE -ne 0){throw 'Could not extract uv.'}
  Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
}
if(!(Test-Path $UvExe)){throw "uv was not found at $UvExe"}
Run $UvExe @('self','version')

Section '2/5 - Local Python + venv'
if(!(Test-Path $PythonExe)){
  Run $UvExe @('python','install',$PythonVersion,'--no-cache','--no-bin','--no-registry')
  Run $UvExe @('venv','--python',$PythonVersion,'--no-cache',$Venv)
}
if(!(Test-Path $PythonExe)){throw "Python venv was not created at $PythonExe"}
$version=(& $PythonExe -c "import sys; print('.'.join(map(str,sys.version_info[:3])))").Trim()
if($LASTEXITCODE -ne 0 -or $version -notmatch '^3\.11\.') {throw "Expected Python 3.11, got '$version'."}

Section '3/5 - Frozen Qwen TTS environment (uv.lock)'
$oldProjectEnvironment = $env:UV_PROJECT_ENVIRONMENT
$env:UV_PROJECT_ENVIRONMENT = $Venv
try {
  Run $UvExe @('sync','--project',$Root,'--frozen','--no-install-project','--no-cache')
} finally {
  $env:UV_PROJECT_ENVIRONMENT = $oldProjectEnvironment
}

Run $PythonExe @('-c',@"
import torch, torchvision, torchaudio
assert torch.__version__.split('+')[0] == '$TorchVersion', torch.__version__
assert torchvision.__version__.split('+')[0] == '0.23.0', torchvision.__version__
assert torchaudio.__version__.split('+')[0] == '$TorchVersion', torchaudio.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
print('Frozen CUDA ABI OK |', torch.__version__, '|', torchvision.__version__, '|', torchaudio.__version__)
"@)

Section '4/5 - Windows FlashAttention 2 build'
$flashBuilder = Join-Path $Root 'flash_attention_installer.ps1'
if (-not (Test-Path -LiteralPath $flashBuilder)) { throw "FlashAttention builder is missing: $flashBuilder" }
try {
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $flashBuilder -Root $Root -UvExe $UvExe -PythonExe $PythonExe
  if ($LASTEXITCODE -ne 0) { throw "FlashAttention builder exited with code $LASTEXITCODE." }
} catch {
  Write-Host "[WARN] FlashAttention $FlashAttentionVersion wheel was not installed: $($_.Exception.Message)" -ForegroundColor Yellow
  Write-Host '[WARN] Qwen TTS remains usable through the built-in SDPA -> eager fallback.' -ForegroundColor Yellow
}

Section '5/5 - Runtime and GUI smoke test'
Run $PythonExe @('-c',@"
import sys, torch, gradio
from qwen_tts import Qwen3TTSModel
from faster_qwen3_tts import FasterQwen3TTS
import peft, qwen_backend
from importlib.metadata import version, PackageNotFoundError
print('Python', sys.version.split()[0])
print('Torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU', torch.cuda.is_available())
print('Gradio', gradio.__version__)
print('Qwen3TTSModel import OK')
print('FasterQwen3TTS import OK | CUDA Graphs backend available for 12 Hz CUDA inference')
print('PEFT import OK')
print('Backend root:', qwen_backend.ROOT)
try:
    print('FlashAttention', version('flash-attn'))
except PackageNotFoundError:
    print('FlashAttention unavailable; runtime fallback remains enabled')
"@)
Write-Host "`n[OK] Installation complete. Models download on demand from the model library or first inference." -ForegroundColor Green
