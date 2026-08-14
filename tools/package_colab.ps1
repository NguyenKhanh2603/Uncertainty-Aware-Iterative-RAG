param(
    [string]$OutputPath = "dist/flare_multihoprag_colab.zip"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ArchivePath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $OutputPath))
$ArchiveDirectory = Split-Path -Parent $ArchivePath

New-Item -ItemType Directory -Force -Path $ArchiveDirectory | Out-Null
if (Test-Path -LiteralPath $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}

Push-Location $RepoRoot
try {
    & tar.exe -a -c -f $ArchivePath `
        --exclude='.git' `
        --exclude='.venv' `
        --exclude='__pycache__' `
        --exclude='.pytest_cache' `
        --exclude='.ruff_cache' `
        --exclude='results' `
        --exclude='results_colab_*' `
        COLAB_TERMINAL_SETUP.md `
        README.md `
        pyproject.toml `
        uv.lock `
        benchmarks/flare_multihoprag `
        data/multihop_rag `
        related_repos/FLARE `
        related_repos/MultiHop-RAG `
        configs `
        eval `
        src `
        tests

    if ($LASTEXITCODE -ne 0) {
        throw "tar.exe failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$Archive = Get-Item -LiteralPath $ArchivePath
Write-Host "Created $($Archive.FullName)"
Write-Host ("Size: {0:N2} MiB" -f ($Archive.Length / 1MB))
