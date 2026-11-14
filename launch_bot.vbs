' ============================================================================
' Crystal Champions Bot Launcher
' ============================================================================
' 
' LAUNCHES: Collector (data) + Panel (overlay) for PRACTICE MODE ONLY
' 
' ⚠️  LEGAL NOTICE:
'    - WorldWinner ToS bans automated play in cash tournaments
'    - Penalties: account termination, forfeiture, legal action
'    - Video Replays are reviewed for bot detection
'    - ONLY USE IN FREE PRACTICE MODE (no cash risk)
' 
' USAGE:
'    1. Open Chrome → WorldWinner → Crystal Champions → PRACTICE mode
'    2. Do NOT start the match yet
'    3. Double-click this .vbs file
'    4. Panel appears in dead-zone left of board
'    5. Click RECOVER → wait for "READY" → click PLAY
'    6. Start match in browser → bot plays automatically
'    7. Press Ctrl+Shift+X to emergency stop
' 
' ============================================================================

Option Explicit

Dim fso, wshShell, pythonPath, projectRoot, cmd, result

Set fso = CreateObject("Scripting.FileSystemObject")
Set wshShell = CreateObject("WScript.Shell")

' Configuration - UPDATE THESE PATHS FOR YOUR SYSTEM
pythonPath = "C:\Users\gurwi\AppData\Local\Programs\Python\Python313\pythonw.exe"
projectRoot = "D:\random works by gs\Crystal Champions helper"

' Verify paths exist
If Not fso.FileExists(pythonPath) Then
    MsgBox "Python not found at: " & pythonPath & vbCrLf & _
           "Edit this .vbs file and update pythonPath variable.", vbCritical, "Path Error"
    WScript.Quit 1
End If

If Not fso.FolderExists(projectRoot) Then
    MsgBox "Project not found at: " & projectRoot & vbCrLf & _
           "Edit this .vbs file and update projectRoot variable.", vbCritical, "Path Error"
    WScript.Quit 1
End If

' Legal confirmation
Dim confirm
confirm = MsgBox( _
    "⚠️  LEGAL WARNING" & vbCrLf & vbCrLf & _
    "WorldWinner ToS bans automated play in cash tournaments." & vbCrLf & _
    "Penalties: account termination, forfeiture, legal action." & vbCrLf & vbCrLf & _
    "This launcher is for PRACTICE MODE ONLY (free, no cash)." & vbCrLf & vbCrLf & _
    "Do you confirm you'll only use this in FREE PRACTICE MODE?", _
    vbYesNo + vbExclamation + vbDefaultButton2, "Legal Confirmation Required")

If confirm <> vbYes Then
    MsgBox "Launch cancelled. Only use in Practice Mode.", vbInformation, "Cancelled"
    WScript.Quit 0
End If

' Change to project directory
wshShell.CurrentDirectory = projectRoot

' Function to run command hidden (no console window)
Function RunHidden(cmdLine)
    On Error Resume Next
    wshShell.Run cmdLine, 0, False  ' 0 = hidden, False = don't wait
    If Err.Number <> 0 Then
        RunHidden = False
        Err.Clear
    Else
        RunHidden = True
    End If
    On Error GoTo 0
End Function

' Start Collector (background data recording)
Dim collectorCmd
collectorCmd = pythonPath & " -m detector.collect --seconds 900 --fps 15 --wait 600"
If RunHidden(collectorCmd) Then
    WScript.Echo "Collector started (15 min, 15fps)"
Else
    MsgBox "Failed to start collector", vbCritical, "Error"
    WScript.Quit 1
End If

' Wait a moment for collector to initialize
WScript.Sleep 3000

' Start Panel (overlay with auto-player)
Dim panelCmd
panelCmd = pythonPath & " -m panel.overlay"
If RunHidden(panelCmd) Then
    WScript.Echo "Panel started (overlay with auto-player)"
Else
    MsgBox "Failed to start panel", vbCritical, "Error"
    WScript.Quit 1
End If

' Success message
MsgBox "✅ Bot Launched Successfully!" & vbCrLf & vbCrLf & _
       "Collector: Recording 15min @ 15fps" & vbCrLf & _
       "Panel: Overlay with auto-player (dead-zone left)" & vbCrLf & vbCrLf & _
       "NEXT STEPS:" & vbCrLf & _
       "1. Open Chrome → Crystal Champions → PRACTICE mode" & vbCrLf & _
       "2. Wait for panel to show 'READY' (shows board ROI)" & vbCrLf & _
       "3. Click RECOVER → wait for 'PLAYING'" & vbCrLf & _
       "4. Click PLAY → start match in browser" & vbCrLf & _
       "5. Bot plays automatically" & vbCrLf & vbCrLf & _
       "KILL SWITCH: Ctrl+Shift+X (instant stop)" & vbCrLf & vbCrLf & _
       "REMEMBER: Practice Mode ONLY!", _
       vbInformation, "Crystal Champions Bot - Ready"