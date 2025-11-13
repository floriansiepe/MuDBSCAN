#!/bin/bash
# Copy a suitable libstdc++.so.6 into ./lib for bundling with a job.
# Usage: ./scripts/bundle_libstdc++.sh /path/to/libstdc++.so.6

if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/libstdc++.so.6"
  exit 1
fi

SRC="$1"
DEST_DIR="$(dirname "$0")/../lib"
mkdir -p "$DEST_DIR"

if [ ! -f "$SRC" ]; then
  echo "Source library not found: $SRC"
  exit 2
fi

cp -v "$SRC" "$DEST_DIR/"
echo "Copied to $DEST_DIR/libstdc++.so.6"
