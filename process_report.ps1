# Load env vars from .env before running (or set them in your shell/system environment)
# Required: OPENAI_API_KEY, ANTHROPIC_API_KEY, HF_API_TOKEN
# $env:CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # fastest Haiku with thinking
# or
$env:CLAUDE_MODEL = "claude-opus-4-1-20250805"   # highest intelligence Opus


python process_reports.py --file reports.json --output test.json --limit 10