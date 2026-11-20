<#
.SYNOPSIS
    Публикация дообученной модели (и при необходимости бинарника) в репозиторий ZESP_asr.

.DESCRIPTION
    Обучение живёт в train/ и в git не попадает. В репозиторий кладутся только
    готовые артефакты: model/model.bin, model/vocab.txt, model/vad.bin и
    bin/asr_zesp_aarch64.

    Скрипт:
      1. экспортирует чекпоинт в формат NNASR1 (train/export_runtime.py);
      2. копирует model.bin/vocab.txt в model/;
      3. с -Vad пересобирает vad.bin (train/export_vad.py);
      4. с -Bin пересобирает C-движок (train/runtime/build_runtime.ps1 -Target aarch64);
      5. печатает md5 и git status.

    Без -Tag ничего не коммитит. С -Tag делает commit + tag + push, после чего
    CI (.github/workflows/release.yml) соберёт архив релиза.

.EXAMPLE
    .\publish.ps1 -Ckpt models\model_domain4.pt
    .\publish.ps1 -Ckpt models\model_domain4.pt -Vad -Tag v0.2.0 -Message "model: domain4"
#>
param(
    [string]$Ckpt = "models\model_domain4.pt",
    [switch]$Vad,
    [switch]$Bin,
    [string]$Tag = "",
    [string]$Message = "",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
$train = Join-Path $repo "train"
$modelDir = Join-Path $repo "model"
$binDir = Join-Path $repo "bin"

if (-not $Python) {
    $Python = Join-Path $train ".venv\Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        $Python = "C:\Users\Vlad\AppData\Local\Programs\Python\Python311\python.exe"
    }
}
if (-not (Test-Path $Python)) { throw "python не найден: $Python" }

$ckptPath = if ([System.IO.Path]::IsPathRooted($Ckpt)) { $Ckpt } else { Join-Path $train $Ckpt }
if (-not (Test-Path $ckptPath)) { throw "чекпоинт не найден: $ckptPath" }

Write-Host "== чекпоинт: $ckptPath"
$tmp = Join-Path $env:TEMP "zesp_asr_publish"
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }

Push-Location $train
try {
    & $Python export_runtime.py --ckpt $ckptPath --out $tmp
    if ($LASTEXITCODE -ne 0) { throw "export_runtime.py завершился с ошибкой" }

    if ($Vad) {
        Write-Host "== VAD: export_vad.py"
        & $Python export_vad.py
        if ($LASTEXITCODE -ne 0) { throw "export_vad.py завершился с ошибкой" }
    }
}
finally {
    Pop-Location
}

New-Item -ItemType Directory -Force $modelDir | Out-Null
Copy-Item (Join-Path $tmp "model.bin") (Join-Path $modelDir "model.bin") -Force
Copy-Item (Join-Path $tmp "vocab.txt") (Join-Path $modelDir "vocab.txt") -Force
if ($Vad) {
    Copy-Item (Join-Path $train "runtime\vad.bin") (Join-Path $modelDir "vad.bin") -Force
}

if ($Bin) {
    Write-Host "== сборка C-движка (aarch64)"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $train "runtime\build_runtime.ps1") -Target aarch64
    if ($LASTEXITCODE -ne 0) { throw "build_runtime.ps1 завершился с ошибкой" }
    New-Item -ItemType Directory -Force $binDir | Out-Null
    Copy-Item (Join-Path $train "runtime\asr_zesp_aarch64") (Join-Path $binDir "asr_zesp_aarch64") -Force
}

Write-Host ""
Write-Host "== артефакты"
foreach ($f in @("model\model.bin", "model\vocab.txt", "model\vad.bin", "bin\asr_zesp_aarch64")) {
    $p = Join-Path $repo $f
    if (Test-Path $p) {
        $h = (Get-FileHash $p -Algorithm MD5).Hash.ToLower()
        $mb = [math]::Round((Get-Item $p).Length / 1MB, 2)
        Write-Host ("  {0,-28} {1,8} MB  {2}" -f $f, $mb, $h)
    }
}

Write-Host ""
Write-Host "== git status"
Push-Location $repo
try {
    git status --short

    if ($Tag) {
        if (-not $Message) { $Message = "model: publish $Tag" }
        git add model bin
        git commit -m $Message
        git tag $Tag
        git push
        git push origin $Tag
        Write-Host "== тег $Tag отправлен, CI соберёт релиз"
    }
    else {
        Write-Host ""
        Write-Host "Дальше (когда решите выпускать):"
        Write-Host "  git add model bin"
        Write-Host "  git commit -m `"model: ...`""
        Write-Host "  git tag v0.2.0"
        Write-Host "  git push; git push origin v0.2.0"
    }
}
finally {
    Pop-Location
}
