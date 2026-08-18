param(
    [switch]$SkipTests,
    [string]$GithubRepository = $env:VISUAL_SEARCH_GITHUB_REPOSITORY
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Missing .venv. Create the development environment first.' }
if (-not $SkipTests -and (Test-Path -LiteralPath 'tests')) {
    & $Python -m pytest
    if ($LASTEXITCODE -ne 0) { throw "Tests failed with exit code $LASTEXITCODE." }
} elseif (-not $SkipTests) {
    Write-Warning 'Local tests are not present in this public checkout; run the release verification suite before publishing.'
}
& $Python scripts\generate-sbom.py --output dist\sbom.json
if ($LASTEXITCODE -ne 0) { throw "SBOM generation failed with exit code $LASTEXITCODE." }
& $Python -m PyInstaller --clean --noconfirm VisualSearch.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE." }
$Iscc = Get-Command iscc.exe -ErrorAction SilentlyContinue
if (-not $Iscc) {
    $Candidates = @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    )
    $Iscc = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $Iscc) { throw 'Inno Setup 6 was not found.' }
& $Iscc installer\VisualSearch.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed with exit code $LASTEXITCODE." }
$Hash = (Get-FileHash -LiteralPath 'dist\VisualSearch-Setup.exe' -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content -LiteralPath 'dist\VisualSearch-Setup.exe.sha256' -Encoding ascii -Value "$Hash  VisualSearch-Setup.exe"
$InstallScript = (Get-Content -LiteralPath 'install.ps1' -Raw).Replace('REPLACE_WITH_RELEASE_SHA256', $Hash)
if ($GithubRepository) {
    if ($GithubRepository -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') {
        throw 'GithubRepository must use owner/repository format.'
    }
    $InstallScript = $InstallScript.Replace('REPLACE_WITH_OWNER/visual-search', $GithubRepository)
}
Set-Content -LiteralPath 'dist\install.ps1' -Encoding utf8 -Value $InstallScript
Write-Host "Done: $ProjectRoot\dist\VisualSearch-Setup.exe"
