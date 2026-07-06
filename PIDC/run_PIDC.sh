#!/bin/bash
set -euo pipefail

INPUT_FILE=$(realpath -- "$1")
OUTPUT_FILE=$(realpath -- "$2")

if [[ -z "$TMPDIR" ]]; then
    TMPDIR="$(mktemp -d)"
fi
SRC_DIR="$(dirname -- "$(realpath -- "$0")")"

python "$SRC_DIR/h5ad_to_tsv.py" --input "$INPUT_FILE" --output "$TMPDIR/data.tsv"
echo "Running PIDC on $INPUT_FILE..."
julia -p 8 "$SRC_DIR/run_PIDC.jl" "$TMPDIR/data.tsv" "$TMPDIR/pidc_output.tsv"
python "$SRC_DIR/tsv_to_csv.py" --input "$TMPDIR/pidc_output.tsv" --output "$OUTPUT_FILE"
echo "PIDC results saved as csv to $OUTPUT_FILE."
