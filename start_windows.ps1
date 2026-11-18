# Run in project root. Requires Python and Node.js 20+.
$ErrorActionPreference = "Stop"
Write-Host "Setup Python packages..."
py -m pip install -r backend/requirements.txt
py scripts/preprocessing/prepare_data.py
if (-not (Test-Path 'models/rossmann_sales_lgbm.joblib')) { py scripts/evaluation/run_evaluation.py }
Write-Host "Starting API and UI..."
Start-Process powershell -ArgumentList '-NoExit','-Command','cd backend; py -m uvicorn app.main:app --reload --port 8000'
Set-Location frontend
npm install
npm run dev
