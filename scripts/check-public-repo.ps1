param([switch]$Staged)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

if (-not (Test-Path -LiteralPath '.git')) {
    throw 'Git repository not initialized. Run this check after git init.'
}
$GitSafety = @('-c', "safe.directory=$ProjectRoot")
$Files = if ($Staged) {
    @(git @GitSafety diff --cached --name-only --diff-filter=ACMR)
} else {
    @(git @GitSafety ls-files --cached --others --exclude-standard)
}
if ($LASTEXITCODE -ne 0) { throw 'Unable to enumerate public files with Git.' }
$Files = @($Files | Where-Object { $_ })
$Errors = [System.Collections.Generic.List[string]]::new()
$Forbidden = '^(\.verification/|\.test-tmp|\.venv/|build/|dist/|release-models/|artifacts/|accuracy-reports/)'
$SensitiveName = '(?i)(^|/)(\.env($|\.)|.*\.(pem|key|pfx|p12|snk|cer|crt|sig)$)'
$MaxBytes = 50MB

foreach ($Relative in $Files) {
    $Normalized = $Relative.Replace('\', '/')
    if ($Normalized -match $Forbidden) { $Errors.Add("forbidden path: $Normalized") }
    if ($Normalized -match $SensitiveName) { $Errors.Add("sensitive filename: $Normalized") }
    $Path = Join-Path $ProjectRoot $Relative
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { continue }
    $Length = (Get-Item -LiteralPath $Path).Length
    if ($Length -gt $MaxBytes) { $Errors.Add("file exceeds 50 MB: $Normalized ($Length bytes)") }
    if ($Length -gt 5MB) { continue }
    $Text = Get-Content -LiteralPath $Path -Raw -ErrorAction SilentlyContinue
    if ($null -eq $Text) { continue }
    $Patterns = @(
        @{ Name = 'private key'; Value = '-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----' },
        @{ Name = 'Hugging Face token'; Value = ('\bhf' + '_[A-Za-z0-9]{20,}\b') },
        @{ Name = 'GitHub token'; Value = ('\bgh' + '[pousr]_[A-Za-z0-9]{20,}\b') },
        @{ Name = 'AWS access key'; Value = '\bAKIA[0-9A-Z]{16}\b' },
        @{ Name = 'Windows user path'; Value = ('[A-Za-z]:' + '\\Users\\[^\\\s"'']+') },
        @{ Name = 'WSL user path'; Value = ('/mnt/[a-z]/' + 'Users/[^/\s"'']+') }
    )
    foreach ($Pattern in $Patterns) {
        if ($Text -match $Pattern.Value) { $Errors.Add("$($Pattern.Name) found in $Normalized") }
    }
}

if ($Errors.Count) {
    $Errors | Sort-Object -Unique | ForEach-Object { Write-Error $_ }
    exit 1
}
Write-Host "Public repository scan passed: $($Files.Count) files, no forbidden paths, secrets, local user paths, or files over 50 MB."
