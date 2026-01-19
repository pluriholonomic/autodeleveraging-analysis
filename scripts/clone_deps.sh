#!/usr/bin/env bash
set -euo pipefail

# Robust script dir + project root resolution
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
EXT_DIR="${ROOT_DIR}/external"

mkdir -p "${EXT_DIR}"

# Git URLs (override with env vars if needed)
HYPERREPLAY_GIT_URL="${HYPERREPLAY_GIT_URL:-https://github.com/ConejoCapital/HyperReplay.git}"
HYPERMULTIADL_GIT_URL="${HYPERMULTIADL_GIT_URL:-https://github.com/ConejoCapital/HyperMultiAssetedADL.git}"

# Pinned commits for reproducibility
# HyperReplay: add-loser-equity-enrichment branch (includes liquidated_total_equity)
HYPERREPLAY_COMMIT="${HYPERREPLAY_COMMIT:-a65fa2295fa98d4aef4db50499e5ba5a3e9fe9c4}"
HYPERMULTIADL_COMMIT="${HYPERMULTIADL_COMMIT:-79bad0fae259fc1fcd9fce960953ae3b398f2db7}"

if [[ ! -d "${EXT_DIR}/HyperReplay" ]]; then
  echo "[clone] HyperReplay -> ${EXT_DIR}/HyperReplay"
  git clone "${HYPERREPLAY_GIT_URL}" "${EXT_DIR}/HyperReplay"
else
  echo "[skip] HyperReplay already exists at ${EXT_DIR}/HyperReplay"
fi

echo "[pin] HyperReplay -> ${HYPERREPLAY_COMMIT}"
git -C "${EXT_DIR}/HyperReplay" fetch --depth 1 origin "${HYPERREPLAY_COMMIT}" >/dev/null 2>&1 || true
git -C "${EXT_DIR}/HyperReplay" checkout -q "${HYPERREPLAY_COMMIT}"

if [[ ! -d "${EXT_DIR}/HyperMultiAssetedADL" ]]; then
  echo "[clone] HyperMultiAssetedADL -> ${EXT_DIR}/HyperMultiAssetedADL"
  git clone "${HYPERMULTIADL_GIT_URL}" "${EXT_DIR}/HyperMultiAssetedADL"
else
  echo "[skip] HyperMultiAssetedADL already exists at ${EXT_DIR}/HyperMultiAssetedADL"
fi

echo "[pin] HyperMultiAssetedADL -> ${HYPERMULTIADL_COMMIT}"
git -C "${EXT_DIR}/HyperMultiAssetedADL" fetch --depth 1 origin "${HYPERMULTIADL_COMMIT}" >/dev/null 2>&1 || true
git -C "${EXT_DIR}/HyperMultiAssetedADL" checkout -q "${HYPERMULTIADL_COMMIT}"

echo "[done] dependencies are under: ${EXT_DIR}"
