#!/bin/bash

# Install PyInstaller if not present
pip install pyinstaller

# Clean previous builds
rm -rf build dist

# Build the binary
python3 -m PyInstaller --clean api.spec

# Output location verification
if [ -f "dist/api" ]; then
    echo "Build complete. The binary is at dist/api"
    echo "To use with Tauri:"
    echo "1. Rename 'dist/api' to 'api-aarch64-apple-darwin' (for Apple Silicon)"
    echo "2. Place it in your Tauri project's src-tauri/binaries/ folder"
    echo "3. Update tauri.conf.json to include the sidecar"
else
    echo "Build failed or binary not found at dist/api"
    exit 1
fi
