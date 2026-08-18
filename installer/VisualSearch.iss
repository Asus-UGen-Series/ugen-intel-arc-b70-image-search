#define MyAppName "Visual Search"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Visual Search"
#define MyAppExeName "VisualSearch.exe"

[Setup]
AppId={{8CB9DD11-E7DA-41A4-A233-534C8C38F03A}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\VisualSearch
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=VisualSearch-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}
LicenseFile=..\LICENSE

[Tasks]
Name: "desktopicon"; Description: "建立桌面捷徑"; GroupDescription: "其他選項："; Flags: unchecked

[Files]
Source: "..\dist\VisualSearch\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dist\sbom.json"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "啟動 {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    if (not UninstallSilent) and
       (MsgBox('是否同時移除 Visual Search 的索引、縮圖、模型與設定？原始圖片不會被刪除。', mbConfirmation, MB_YESNO) = IDYES) then
      DelTree(ExpandConstant('{localappdata}\VisualSearch'), True, True, True);
end;
