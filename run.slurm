#!/bin/bash
#SBATCH --job-name=MuDBSCANJob
#SBATCH --nodes=4
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --output=mu-dbscan_output_%j.txt
#SBATCH --time=01:00:00
#SBATCH --mail-user=siepef@uni-marburg.de
#SBATCH --mail-type=END
#SBATCH --mem=128G
##SBATCH --exclusive
# You can override partition, qos, account etc. by editing the header or
# passing them to sbatch on the command line. Adjust nodes/ntasks-per-node
# to match the resources you need.

#############################################################################
# Usage
# sbatch run.slurm <dataset> <minEntries> <eps> <minPts> <num_partitions> <exp_dir> <out>
#
# Notes:
# - <minEntries> will be used for both MinEntries and (by default) MaxEntries*2
#   unless you edit this file to provide explicit MaxEntries.
# - The script expects the `output` binary to be in the submission directory.
# - The script runs the binary via srun. If your cluster prefers mpirun, edit
#   the invocation accordingly.
#############################################################################

DATASET="$1"
MINENTRIES="$2"
EPS="$3"
MINPTS="$4"
NUM_PARTITIONS="$5"
EXP_DIR="$6"

if [ -z "$DATASET" ] || [ -z "$MINENTRIES" ] || [ -z "$EPS" ] || [ -z "$MINPTS" ] || [ -z "$NUM_PARTITIONS" ] || [ -z "$EXP_DIR" ]; then
  echo "Usage: sbatch run.slurm <dataset> <minEntries> <eps> <minPts> <num_partitions> <exp_dir>"
  exit 1
fi

# Basic mapping: the C++ program expects: <InputPath> <Epsilon> <MINPOINTS> <MinEntries> <MaxEntries> <OutputFile>
# We use MINENTRIES as provided and set MAXENTRIES = MINENTRIES * 2 by default. Edit if you want other values.
MAXENTRIES=$(( MINENTRIES * 2 ))

# Change to submit directory where binary lives if available
if [ -n "$SLURM_SUBMIT_DIR" ]; then
  cd "$SLURM_SUBMIT_DIR" || exit 1
fi

echo "Starting μDBSCAN job"
echo "Job id: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Requested ranks: $NUM_PARTITIONS"
echo "Dataset: $DATASET"
echo "Eps: $EPS  MinPts: $MINPTS  MinEntries: $MINENTRIES  MaxEntries: $MAXENTRIES"


module purge
module load openmpi4

# If a bundled lib or lib64 directory exists in the submit directory, prepend it
# to LD_LIBRARY_PATH so bundled libstdc++.so.6 is used on compute nodes.
if [ -d "${SLURM_SUBMIT_DIR:-.}/lib" ]; then
  export LD_LIBRARY_PATH="${SLURM_SUBMIT_DIR:-.}/lib:${LD_LIBRARY_PATH:-}"
  echo "Prepending ${SLURM_SUBMIT_DIR:-.}/lib to LD_LIBRARY_PATH"
elif [ -d "${SLURM_SUBMIT_DIR:-.}/lib64" ]; then
  export LD_LIBRARY_PATH="${SLURM_SUBMIT_DIR:-.}/lib64:${LD_LIBRARY_PATH:-}"
  echo "Prepending ${SLURM_SUBMIT_DIR:-.}/lib64 to LD_LIBRARY_PATH"
fi

# Prepare a per-job scratch output file (so the compute nodes write to shared scratch)
# Default scratch base directory can be overridden by setting SCRATCH_BASE in the environment
SCRATCH_BASE="${SCRATCH_BASE:-/scratch_shared/siepef}"
SCRATCH_FILE="$SCRATCH_BASE/scratch-MuDBSCAN-${SLURM_JOB_ID}"

# Run using srun (preferred on Slurm). Adjust --mpi or replace with mpirun if required.
# The program's output file argument now points to the per-job scratch file.
srun --mpi=pmix --nodes=$SLURM_NNODES --ntasks=$NUM_PARTITIONS ./output "$DATASET" "$EPS" "$MINPTS" "$MINENTRIES" "$MAXENTRIES" "$SCRATCH_FILE"

RET=$?
echo "run finished with exit code $RET"
  
# Attempt to parse the run log and generate metrics.json using parse_out.sh
PARSE_SCRIPT="${SLURM_SUBMIT_DIR:-.}/parse_out.sh"
PARSE_LOG="$EXP_DIR/run_${SLURM_JOB_ID}.log"
if [ -x "$PARSE_SCRIPT" ]; then
  echo "Parsing run log with $PARSE_SCRIPT -> $PARSE_LOG"
  "$PARSE_SCRIPT" "$PARSE_LOG" "$EXP_DIR" || echo "Warning: parse_out.sh failed" >&2
else
  echo "parse_out.sh not found or not executable at $PARSE_SCRIPT; skipping metrics generation" >&2
fi

exit $RET
