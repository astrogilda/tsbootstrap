#!/usr/bin/env bash
# Run a SonarCloud analysis from a developer machine, using the same project key,
# organization, and exclusions that CI uses (read from sonar-project.properties).
#
# CI uses SonarCloud automatic analysis, which has no merge-gating power here; this
# script lets you reproduce the same scan locally before pushing so a new bug or
# security finding does not surprise you on the PR.
#
# Requirements:
#   - SONAR_TOKEN in the environment (a SonarCloud user token with analysis rights).
#     Generate one at https://sonarcloud.io/account/security and export it, or store
#     it in the keyring and load it, for example:
#       export SONAR_TOKEN="$(secret-tool lookup service sonarcloud account token)"
#   - Either the sonar-scanner CLI on PATH, or Docker (the script falls back to the
#     official sonarsource/sonar-scanner-cli image).
#
# Coverage: pass a coverage XML path as the first argument to include coverage in
# the scan, for example:
#   uv run python -m pytest tests/ --cov=src/tsbootstrap --cov-report=xml
#   scripts/sonar-local.sh coverage.xml
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ -z "${SONAR_TOKEN:-}" ]; then
  echo "error: SONAR_TOKEN is not set. See the header of this script for how to provide it." >&2
  exit 1
fi

COVERAGE_ARG=()
if [ -n "${1:-}" ]; then
  if [ ! -f "$1" ]; then
    echo "error: coverage file '$1' not found." >&2
    exit 1
  fi
  COVERAGE_ARG=(-Dsonar.python.coverage.reportPaths="$1")
fi

# A local scan analyzes the working tree on the current branch, not a pull request.
COMMON_ARGS=(
  -Dsonar.host.url=https://sonarcloud.io
  -Dsonar.branch.name="$(git rev-parse --abbrev-ref HEAD)"
  "${COVERAGE_ARG[@]}"
)

if command -v sonar-scanner >/dev/null 2>&1; then
  exec sonar-scanner -Dsonar.token="${SONAR_TOKEN}" "${COMMON_ARGS[@]}"
elif command -v docker >/dev/null 2>&1; then
  exec docker run --rm \
    -e SONAR_TOKEN="${SONAR_TOKEN}" \
    -v "$(pwd):/usr/src" \
    sonarsource/sonar-scanner-cli "${COMMON_ARGS[@]}"
else
  echo "error: neither sonar-scanner nor docker is available on PATH." >&2
  exit 1
fi
