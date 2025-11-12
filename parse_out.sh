#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <out_file> [base_dir]"
  echo "  out_file   : path to the program output file (the 'out' you showed)"
  echo "  base_dir   : root directory where dataset folders will be created (default: .)"
  exit 1
fi

OUTFILE="$1"
BASEDIR="${2:-.}"

if [ ! -f "$OUTFILE" ]; then
  echo "Error: file not found: $OUTFILE" >&2
  exit 2
fi

# Find the runtime line (TOTAL without file IO). The original output uses the misspelled
# label "Total [Wihtout File IO]" — be permissive and look for "Total" and "File IO".
time_line=$(grep -m1 -E "Total.*File IO" "$OUTFILE" || true)
if [ -z "$time_line" ]; then
  # try a fallback for other possible labels
  time_line=$(grep -m1 -i "Total \[without file io\]" "$OUTFILE" || true)
fi

if [ -z "$time_line" ]; then
  echo "Error: could not find a 'Total ... File IO' line in $OUTFILE" >&2
  exit 3
fi

# Extract seconds (a floating point number) from the matching line
time_sec=$(echo "$time_line" | sed -E 's/.*:[[:space:]]*([0-9]+\.?[0-9]*).*/\1/')
if [ -z "$time_sec" ]; then
  echo "Error: failed to parse seconds from: $time_line" >&2
  exit 4
fi

# Convert to milliseconds (rounded)
algoTimeMs=$(awk -v s="$time_sec" 'BEGIN{printf("%.0f", s*1000)}')

# Extract Eps and MinPts from a line that contains them (e.g. "Eps: 0.025 Minpts: 10")
eps=$(grep -m1 -E "Eps:" "$OUTFILE" | awk '{for(i=1;i<=NF;i++){ if($i=="Eps:") {print $(i+1); exit}}}') || true
minpts=$(grep -m1 -E "Minpts:" "$OUTFILE" | awk '{for(i=1;i<=NF;i++){ if($i=="Minpts:") {print $(i+1); exit}}}') || true

if [ -z "$eps" ] || [ -z "$minpts" ]; then
  # fallback: try case-insensitive search
  eps=$(grep -mi1 -E "Eps:" "$OUTFILE" | awk '{for(i=1;i<=NF;i++){ if(tolower($i)=="eps:") {print $(i+1); exit}}}') || true
  minpts=$(grep -mi1 -E "Minpts:" "$OUTFILE" | awk '{for(i=1;i<=NF;i++){ if(tolower($i)=="minpts:") {print $(i+1); exit}}}') || true
fi

if [ -z "$eps" ] || [ -z "$minpts" ]; then
  echo "Warning: could not parse Eps/Minpts from $OUTFILE; defaulting to eps=unknown minPts=0" >&2
  eps="unknown"
  minpts=0
fi

# Extract the Filename token to infer dataset name (line like: "Filename: /path/to/densired_2_shrink.csv Eps: ...")
filename_token=$(grep -m1 -E "Filename:" "$OUTFILE" || true)
inputpath=""
if [ -n "$filename_token" ]; then
  inputpath=$(echo "$filename_token" | sed -E 's/.*Filename:[[:space:]]*([^[:space:]]+).*/\1/')
fi

dataset=""
if [ -n "$inputpath" ]; then
  basefn=$(basename "$inputpath")
  name_noext=${basefn%.*}
  # If the name looks like densired_2_shrink take densired_2; otherwise use the full base name
  if [[ "$name_noext" =~ ^(densired_[0-9]+) ]]; then
    dataset=${BASH_REMATCH[1]}
  else
    dataset="$name_noext"
  fi
else
  dataset="unknown_dataset"
fi

# Clean up eps string: remove trailing zeros and trailing decimal point (e.g. 0.050000 -> 0.05)
eps_clean="$eps"
if [ "$eps" != "unknown" ]; then
  eps_clean=$(awk -v v="$eps" 'BEGIN{ if(v=="" || (v+0)!=v) { print v } else { printf("%g", v) } }')
fi

# Prepare target dir and metrics file path
target_dir="$BASEDIR/$dataset/MuDBSCAN/${eps_clean}_${minpts}"
mkdir -p "$target_dir"
metrics_file="$target_dir/metrics.json"

# Write JSON (pretty simple formatting)
cat > "$metrics_file" <<EOF
{
  "algo" : "MuDBSCAN",
  "algoTimeMs" : $algoTimeMs,
  "clusterParameters" : {
    "eps" : $eps_clean,
    "minPts" : $minpts
  },
  "datasetParameters" : {
    "datasetName" : "$dataset"
  }
}
EOF

echo "Wrote $metrics_file (algoTimeMs=$algoTimeMs ms)"
exit 0
