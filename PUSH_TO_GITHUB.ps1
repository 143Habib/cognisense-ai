$ErrorActionPreference = "Stop"
$RepoUrl = "https://github.com/143Habib/cognisense-ai.git"
$RepoName = "143Habib/cognisense-ai"

Write-Host "`nCOGNISENSE AI / GitHub publisher" -ForegroundColor Green
Write-Host "Target: $RepoName`n" -ForegroundColor Cyan

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git for Windows is required. Opening the official installer page..." -ForegroundColor Yellow
    Start-Process "https://git-scm.com/download/win"
    throw "Install Git for Windows, then double-click PUSH_TO_GITHUB.bat again."
}

if (-not (Test-Path "README.md") -or -not (Test-Path "server.py")) {
    throw "Run this publisher from the extracted CogniSense_AI_GitHub_Ready project folder."
}

if (-not (Test-Path ".git")) { git init }
git branch -M main

$existing = git remote get-url origin 2>$null
if ($LASTEXITCODE -eq 0 -and $existing) {
    if ($existing -ne $RepoUrl) { git remote set-url origin $RepoUrl }
} else {
    git remote add origin $RepoUrl
}

# Never publish MAUS raw recordings or the curated raw subset by accident.
foreach ($datasetPath in @("data/MAUS.zip", "data/maus_research_subset.zip")) {
    if (Test-Path $datasetPath) {
        git check-ignore $datasetPath | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Safety check failed: $datasetPath is not ignored. Stop; do not publish raw dataset archives." }
    }
}

$remoteMain = git ls-remote --heads origin main
if ($LASTEXITCODE -ne 0) { throw "Could not access GitHub. Sign in through Git Credential Manager and retry." }
if ($remoteMain) { throw "The GitHub main branch is not empty. This safety script will not overwrite existing remote history." }

if (-not (git config user.name)) { git config user.name "143Habib" }
if (-not (git config user.email)) { git config user.email "143Habib@users.noreply.github.com" }

git add -A
$changes = git status --porcelain
if (-not $changes) { throw "No project files were found to publish." }
Write-Host "Files staged for the initial commit:" -ForegroundColor Cyan
git status --short

git commit -m "Initial release: CogniSense AI multimodal research observatory"
git push -u origin main

Write-Host "`nPublished successfully: https://github.com/143Habib/cognisense-ai" -ForegroundColor Green
Start-Process "https://github.com/143Habib/cognisense-ai"
