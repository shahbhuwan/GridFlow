[Setup]
AppName=GridFlow
AppVersion=1.0
AppPublisher=Bhuwan Shah
AppCopyright=Copyright (C) 2025 Bhuwan Shah
DefaultDirName={localappdata}\Programs\GridFlow
DefaultGroupName=GridFlow
OutputDir=dist
OutputBaseFilename=GridFlowGUI_Setup
SetupIconFile=gridflow_logo.ico
LicenseFile=LICENSE.txt
Compression=lzma
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName=GridFlow
UninstallDisplayIcon={app}\gridflow.exe
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Files]
Source: "dist\gridflow\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "gridflow_logo.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{userdesktop}\GridFlow"; Filename: "{app}\gridflow.exe"; IconFilename: "{app}\gridflow_logo.ico"
Name: "{userstartmenu}\GridFlow"; Filename: "{app}\gridflow.exe"; IconFilename: "{app}\gridflow_logo.ico"
Name: "{userstartmenu}\{cm:UninstallProgram,GridFlow}"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\gridflow.exe"; Description: "{cm:LaunchProgram,GridFlow}"; Flags: nowait postinstall skipifsilent