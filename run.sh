#!/usr/bin/env bash
# Get dataset, minEntries, eps, minPts, num_partitions and exp_dir from command line arguments
DATASET=${1}
EPS=${3}
MINPTS=${4}
NUM_PARTITIONS=${5}
EXP_DIR=${6}
MINENTRIES=100

# Check if required arguments are provided
if [ -z "$DATASET" ] || [ -z "$EPS" ] || [ -z "$MINPTS" ] || [ -z "$NUM_PARTITIONS" ] || [ -z "$EXP_DIR" ]; then
  echo "Usage: $0 <dataset> <minEntries> <eps> <minPts> <num_partitions> <exp_dir> [out]"
  exit 1
fi

# For pure-MPI runs set NUM_PARTITIONS to number of ranks (e.g. 128 for 4 nodes × 32 cores)
# Forward NUM_PARTITIONS as the ntasks value to sbatch so the allocation matches the run.
sbatch run.slurm "$DATASET" "$MINENTRIES" "$EPS" "$MINPTS" "$NUM_PARTITIONS" "$EXP_DIR"

exit 0


