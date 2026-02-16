#!/usr/bin/env powershell
# Reset the stuck rebase
Set-Location 'D:\Codes\StockSense'

# First, try to clean up the rebase-merge directory
if (Test-Path '.git/rebase-merge') {
    Remove-Item -Recurse -Force '.git/rebase-merge' -ErrorAction SilentlyContinue
    Write-Host "Cleaned up rebase-merge directory"
}

# Reset to ORIG_HEAD to undo the rebase
git reset --hard ORIG_HEAD
if ($LASTEXITCODE -eq 0) {
    Write-Host "Successfully reset to ORIG_HEAD"
} else {
    Write-Host "Reset failed"
}

# Now do a merge instead of rebase (simpler for conflict resolution)
Write-Host "Pulling with merge strategy..."
git pull origin main --no-rebase

Write-Host "Final status:"
git status
