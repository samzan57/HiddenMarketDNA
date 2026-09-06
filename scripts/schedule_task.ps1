# schedule_task.ps1
# Enregistre le live trader dans Windows Task Scheduler.
# USAGE : exécuter en tant qu'Administrateur dans PowerShell, depuis n'importe où
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\scripts\schedule_task.ps1

$TaskName    = "HiddenMarketDNA_LiveTrader"
$ProjectDir  = (Resolve-Path "$PSScriptRoot\..").Path
$BatchScript = "$PSScriptRoot\run_live_trader.bat"

# Lundi à 15h35 UTC = 9h35 NY (heure été EST+1)
# Task Scheduler utilise l'heure locale — adapter si besoin
$TriggerTime = "15:35"

# Supprimer l'ancienne tâche si elle existe
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

# Déclencheur : chaque lundi
$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday `
    -At $TriggerTime

# Action : lancer le batch
$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatchScript`"" `
    -WorkingDirectory $ProjectDir

# Paramètres : continuer même si l'utilisateur n'est pas connecté
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 10) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

# Enregistrement
Register-ScheduledTask `
    -TaskName $TaskName `
    -Trigger $Trigger `
    -Action $Action `
    -Settings $Settings `
    -Description "HiddenMarketDNA — Live trading hebdomadaire (lundi 15h35 UTC)" `
    -RunLevel Highest

Write-Host ""
Write-Host "Tâche '$TaskName' enregistrée avec succès." -ForegroundColor Green
Write-Host "Prochain lundi à $TriggerTime UTC, le live trader se lancera automatiquement."
Write-Host ""
Write-Host "Commandes utiles :"
Write-Host "  Voir la tâche     : Get-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Tester maintenant : Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Supprimer         : Unregister-ScheduledTask -TaskName '$TaskName'"
