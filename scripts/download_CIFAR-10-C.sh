#!/usr/bin/env bash
# Downloads and extracts the CIFAR-10-C dataset into data/

set -euo pipefail

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
ARCHIVE="$DEST_DIR/CIFAR-10-C.tar"
URL="https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"

mkdir -p "$DEST_DIR"

echo "Downloading CIFAR-10-C dataset..."
curl -L --progress-bar -o "$ARCHIVE" "$URL"

echo "Extracting archive to $DEST_DIR ..."
tar -xf "$ARCHIVE" -C "$DEST_DIR"

echo "Cleaning up archive..."
rm "$ARCHIVE"

echo "Done. Dataset available at $DEST_DIR/CIFAR-10-C"

