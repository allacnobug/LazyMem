#!/usr/bin/env bash
# Check or apply the VERL patches required by LazyMem reward aggregation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-apply}"

if [[ "$MODE" != "apply" && "$MODE" != "--check" ]]; then
  echo "usage: $0 [--check]" >&2
  exit 2
fi

SITE_PACKAGES="$(
  "$PYTHON_BIN" -c 'from pathlib import Path; import verl; print(Path(verl.__file__).resolve().parent.parent)'
)"

apply_one() {
  local patch_file="$1"
  if patch --dry-run --forward -p1 -d "$SITE_PACKAGES" < "$patch_file" >/dev/null 2>&1; then
    if [[ "$MODE" == "--check" ]]; then
      echo "applicable: $(basename "$patch_file")"
    else
      patch --forward -p1 -d "$SITE_PACKAGES" < "$patch_file"
    fi
    return
  fi

  if patch --dry-run --reverse -p1 -d "$SITE_PACKAGES" < "$patch_file" >/dev/null 2>&1; then
    echo "already applied: $(basename "$patch_file")"
    return
  fi

  echo "cannot apply cleanly: $patch_file" >&2
  exit 1
}

apply_one "$REPO_ROOT/patches/verl-0.7.1/metric_utils.patch"
apply_one "$REPO_ROOT/patches/verl-0.7.1/ray_trainer.patch"
