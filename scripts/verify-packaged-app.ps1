param([string]$Application = 'dist\VisualSearch\VisualSearch.exe')
$ErrorActionPreference = 'Stop'
$Application = (Resolve-Path -LiteralPath $Application).Path
$AuditRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('VisualSearch-audit-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $AuditRoot | Out-Null
$PreviousData = $env:VISUAL_SEARCH_DATA_DIR
try {
    $env:VISUAL_SEARCH_DATA_DIR = $AuditRoot
    $Process = Start-Process -FilePath $Application -ArgumentList '--diagnose' -WindowStyle Hidden -PassThru
    if (-not $Process.WaitForExit(120000)) {
        throw "Packaged diagnosis timed out. Test process id: $($Process.Id)"
    }
    if ($Process.ExitCode -ne 0) { throw "Packaged diagnosis failed: $($Process.ExitCode)" }
    $ReportPath = Join-Path $AuditRoot 'logs\latest-diagnostics.json'
    $Report = Get-Content -LiteralPath $ReportPath -Raw -Encoding utf8 | ConvertFrom-Json
    if ($Report.indexed_images -ne 0) { throw 'Expected an empty isolated library.' }
    if ([IO.Path]::GetFullPath($Report.application_data) -ne $AuditRoot) { throw 'Data isolation failed.' }
    Write-Host "PASS: bundled runtime and empty-library diagnosis. Report: $ReportPath"
    Write-Host 'This is not a clean Windows installation or a GPU inference acceptance test.'
} finally {
    $env:VISUAL_SEARCH_DATA_DIR = $PreviousData
}
