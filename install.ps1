param(
    [string]$Version = "1.0.0",
    [string]$BaseUrl = "https://github.com/REPLACE_WITH_OWNER/visual-search/releases/download",
    [string]$ExpectedSha256 = "REPLACE_WITH_RELEASE_SHA256"
)
$ErrorActionPreference = 'Stop'
if ($BaseUrl -match 'REPLACE_WITH' -or $ExpectedSha256 -match 'REPLACE_WITH') {
    throw '發行者尚未設定正式下載網址與 SHA-256；請直接執行 VisualSearch-Setup.exe。'
}
$DownloadDir = Join-Path $env:TEMP 'VisualSearchInstaller'
New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null
$Installer = Join-Path $DownloadDir "VisualSearch-$Version-Setup.exe"
$Url = "$BaseUrl/v$Version/VisualSearch-Setup.exe"
Invoke-WebRequest -Uri $Url -OutFile $Installer -UseBasicParsing
$Actual = (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash
if ($Actual -ne $ExpectedSha256) { Remove-Item -LiteralPath $Installer -Force; throw '安裝程式驗證失敗，已停止安裝。' }
$Signature = Get-AuthenticodeSignature -LiteralPath $Installer
if ($Signature.Status -notin @('Valid', 'NotSigned')) { throw "數位簽章驗證失敗：$($Signature.Status)" }
Start-Process -FilePath $Installer -Wait
