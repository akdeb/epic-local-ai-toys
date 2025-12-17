#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCH="$(uname -m)"
case "$ARCH" in
  arm64) TARGET_TRIPLE="aarch64-apple-darwin" ;;
  x86_64) TARGET_TRIPLE="x86_64-apple-darwin" ;;
  *)
    echo "Unsupported architecture: $ARCH" >&2
    exit 1
    ;;
esac

SIDE_SRC_BIN="$ROOT_DIR/dist/api"
TAURI_BIN_DIR="$ROOT_DIR/elato-ui/src-tauri/binaries"
TAURI_DEST_BIN="$TAURI_BIN_DIR/api-$TARGET_TRIPLE"

# ============================================================
# 1. Kill any existing API on :8000
# ============================================================
echo "[dev_tauri] Stopping any existing API on :8000..."
curl -s -X POST http://127.0.0.1:8000/shutdown >/dev/null 2>&1 || true
sleep 0.5
# Force kill anything still on 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# ============================================================
# 2. Nuke ALL caches so PyInstaller builds fresh
# ============================================================
echo "[dev_tauri] Nuking build caches..."
rm -rf "$ROOT_DIR/build"
rm -rf "$ROOT_DIR/dist"
rm -rf "$TAURI_DEST_BIN"
rm -rf ~/Library/Application\ Support/pyinstaller/bincache* 2>/dev/null || true
find "$ROOT_DIR" -maxdepth 3 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# IMPORTANT:
# Don't delete src-tauri/target/**/api. During `tauri dev`, Tauri may spawn the cached
# sidecar at src-tauri/target/debug/api. If we delete it and Cargo doesn't rebuild,
# the app will panic with "No such file or directory".

# ============================================================
# 3. Build sidecar with PyInstaller (fresh)
# ============================================================
echo "[dev_tauri] Building sidecar (PyInstaller)..."
(
  cd "$ROOT_DIR"
  python3 -m PyInstaller --clean --noconfirm api.spec
)

if [ ! -f "$SIDE_SRC_BIN" ]; then
  echo "[dev_tauri] ERROR: Sidecar not found at $SIDE_SRC_BIN" >&2
  exit 1
fi

# ============================================================
# 4. Verify the binary has latest code
# ============================================================
echo "[dev_tauri] Verifying sidecar build..."
BUILD_INFO=$("$SIDE_SRC_BIN" --print-build-info 2>&1) || true
echo "$BUILD_INFO"
if [[ "$BUILD_INFO" != *"ELATO_BUILD_INFO"* ]]; then
  echo "[dev_tauri] WARNING: Binary may be stale (no build info found)"
fi

# ============================================================
# 5. Install sidecar for Tauri
# ============================================================
echo "[dev_tauri] Installing sidecar -> $TAURI_DEST_BIN"
mkdir -p "$TAURI_BIN_DIR"
cp "$SIDE_SRC_BIN" "$TAURI_DEST_BIN"
chmod +x "$TAURI_DEST_BIN"

# Tauri dev commonly spawns the sidecar from src-tauri/target/debug/api.
# Force-overwrite it so Tauri can't accidentally run an old cached binary.
TAURI_DEBUG_SIDECAR="$ROOT_DIR/elato-ui/src-tauri/target/debug/api"
mkdir -p "$(dirname "$TAURI_DEBUG_SIDECAR")"
cp "$SIDE_SRC_BIN" "$TAURI_DEBUG_SIDECAR"
chmod +x "$TAURI_DEBUG_SIDECAR"

# ============================================================
# 6. Run Tauri dev
# ============================================================
echo "[dev_tauri] Starting Tauri dev..."
cd "$ROOT_DIR/elato-ui"
npm run tauri dev
