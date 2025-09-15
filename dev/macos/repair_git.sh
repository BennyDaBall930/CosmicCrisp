#!/usr/bin/env bash
# Repair noisy source-control status locally (permissions/line endings/large vendor tree)
set -euo pipefail

echo "Configuring git to ignore chmod noise and CRLF conversions..."
git config core.filemode false
git config core.autocrlf input

echo "Reverting vendored searxng tree to HEAD..."
git restore --staged --worktree --source=HEAD -- searxng || true

echo "Reverting other tracked files under logs/ that may have been touched..."
git restore --staged --worktree --source=HEAD -- logs || true

echo "Preview of remaining changes (if any):"
git status --short

cat <<NOTE

If you still see a lot of untracked files (run/, tmp/, logs/*.log), you can clean them:
  git clean -nXdf   # preview ignored files to be removed
  git clean -Xdf    # remove ignored files

To nuke every untracked file (USE WITH CARE):
  git clean -nfdx   # preview
  git clean -fdx    # execute

NOTE

