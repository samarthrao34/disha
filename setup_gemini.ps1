# DISHA - Google Gemini Flash Setup Script
# This script will set up DISHA with Gemini Flash API

Write-Host "`n" -NoNewline
Write-Host "="*70 -ForegroundColor Cyan
Write-Host "  🌟 DISHA - Google Gemini Flash Setup 🌟" -ForegroundColor Yellow
Write-Host "="*70 -ForegroundColor Cyan
Write-Host "`n"

Write-Host "Setting up DISHA with Google Gemini Flash..." -ForegroundColor Green
Write-Host "Free tier: 1,500 requests/day - Perfect for startups!`n" -ForegroundColor White

# Step 1: Activate virtual environment
Write-Host "[1/4] Activating virtual environment..." -ForegroundColor Cyan
& ".\.venv\Scripts\Activate.ps1"

# Step 2: Install Gemini package
Write-Host "`n[2/4] Installing Google Generative AI package..." -ForegroundColor Cyan
pip install google-generativeai

# Step 3: Check for API key
Write-Host "`n[3/4] Checking API key..." -ForegroundColor Cyan
$envContent = Get-Content -Path ".env" -Raw

if ($envContent -match "GEMINI_API_KEY=your_gemini_api_key_here" -or $envContent -match "GEMINI_API_KEY=\s*$") {
    Write-Host "`n⚠️  No valid Gemini API key found!" -ForegroundColor Yellow
    Write-Host "`nTo get your FREE API key:" -ForegroundColor White
    Write-Host "1. Go to: https://makersuite.google.com/app/apikey" -ForegroundColor White
    Write-Host "2. Click 'Create API Key'" -ForegroundColor White
    Write-Host "3. Copy your key" -ForegroundColor White
    Write-Host "4. Add it to .env file: GEMINI_API_KEY=your_key_here`n" -ForegroundColor White
    
    $apiKey = Read-Host "Paste your Gemini API key here (or press Enter to add it manually later)"
    
    if ($apiKey -ne "") {
        $envContent = $envContent -replace "GEMINI_API_KEY=.*", "GEMINI_API_KEY=$apiKey"
        Set-Content -Path ".env" -Value $envContent
        Write-Host "✅ API key saved to .env file!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Remember to add your API key to .env file before running DISHA!" -ForegroundColor Yellow
    }
} else {
    Write-Host "✅ Gemini API key found in .env file!" -ForegroundColor Green
}

# Step 4: Done!
Write-Host "`n[4/4] Setup complete!" -ForegroundColor Cyan

Write-Host "`n" -NoNewline
Write-Host "="*70 -ForegroundColor Cyan
Write-Host "  ✅ DISHA is ready to use with Gemini Flash!" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Cyan
Write-Host "`n"

Write-Host "To start DISHA:" -ForegroundColor White
Write-Host "  python disha_minimal.py" -ForegroundColor Cyan
Write-Host "`nOr use the launcher:" -ForegroundColor White
Write-Host "  .\START_DISHA.ps1" -ForegroundColor Cyan
Write-Host "`n"

Write-Host "Features:" -ForegroundColor Yellow
Write-Host "  ✅ Ultra-fast responses (sub-second)" -ForegroundColor White
Write-Host "  ✅ 1,500 requests/day FREE" -ForegroundColor White
Write-Host "  ✅ Fine-tunable for your needs" -ForegroundColor White
Write-Host "  ✅ Commercially licensed" -ForegroundColor White
Write-Host "  ✅ Perfect for startups!" -ForegroundColor White
Write-Host "`n"
