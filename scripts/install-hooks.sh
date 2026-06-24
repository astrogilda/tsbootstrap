#!/usr/bin/env bash
# Point this clone at the version-controlled hooks in .githooks/ so every
# contributor runs the same pre-push gate. Idempotent: safe to run repeatedly.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath .githooks
echo "core.hooksPath set to .githooks; pre-push gate is active for this clone."
