# DISHA Quick Launcher
# Double-click this file to run DISHA!

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "          🌟 DISHA - Quick Launcher 🌟" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (Test-Path ".\.venv\Scripts\python.exe") {
    $python = ".\.venv\Scripts\python.exe"
} else {
    $python = "python"
}

Write-Host "Choose DISHA version:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Minimal DISHA (RECOMMENDED - Fast, Always Works)" -ForegroundColor Green
Write-Host "   - Instant startup"
Write-Host "   - Text input, voice output"
Write-Host "   - Perfect for daily use"
Write-Host ""
Write-Host "2. Full DISHA (Advanced Features)" -ForegroundColor Cyan
Write-Host "   - Voice input OR text input"
Write-Host "   - Memory & emotion detection"
Write-Host "   - Slower startup"
Write-Host ""

$choice = Read-Host "Enter choice (1 or 2)"

if ($choice -eq "1") {
    Write-Host ""
    Write-Host "Starting Minimal DISHA..." -ForegroundColor Green
    Write-Host ""
    & $python disha_minimal.py
} elseif ($choice -eq "2") {
    Write-Host ""
    Write-Host "Starting Full DISHA..." -ForegroundColor Cyan
    Write-Host "(This may take 10-30 seconds to load...)" -ForegroundColor Yellow
    Write-Host ""
    & $python DISHAMemory.py
} else {
    Write-Host ""
    Write-Host "Invalid choice. Please run again and choose 1 or 2." -ForegroundColor Red
    Write-Host ""
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
