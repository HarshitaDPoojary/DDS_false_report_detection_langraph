# run_incident_severity.ps1
# (optional) activate your venv if needed:
# . .\.venv\Scripts\Activate.ps1   # or . .\test\Scripts\Activate.ps1

# Load env vars from .env before running (or set them in your shell/system environment)
# Required: OPENAI_API_KEY, ANTHROPIC_API_KEY, HF_API_TOKEN
# $env:CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # fastest Haiku with thinking
# or
$env:CLAUDE_MODEL = "claude-opus-4-1-20250805"   # highest intelligence Opus
$env:OPENAI_MODEL = "gpt-4.1-mini"


python incident_severity_score.py   # change to your entrypoint (e.g., incident_severity.py)