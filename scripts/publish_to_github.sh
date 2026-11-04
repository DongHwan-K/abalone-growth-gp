#!/usr/bin/env bash
set -euo pipefail

OWNER="${GITHUB_OWNER:-DongHwan-K}"
REPOSITORY="${GITHUB_REPOSITORY_NAME:-abalone-growth-gp}"
VISIBILITY="${1:-public}"

if [[ "$VISIBILITY" != "public" && "$VISIBILITY" != "private" ]]; then
  echo "Usage: $0 [public|private]" >&2
  exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is required: https://cli.github.com/" >&2
  exit 1
fi

gh auth status

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python scripts/check_release.py

if [[ ! -d .git ]]; then
  git init -b main
fi

git add .
if ! git diff --cached --quiet; then
  git commit -m "Initial public release for the abalone growth prediction paper"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin main
else
  gh repo create "$OWNER/$REPOSITORY" "--$VISIBILITY" --source=. --remote=origin --push \
    --description "Reproducibility code and results for stacked monotone Gaussian process prediction of Pacific abalone nursery growth."
fi

echo "Published: https://github.com/$OWNER/$REPOSITORY"
