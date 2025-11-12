# MuDBSCAN

A fast, exact, and scalable algorithm for DBSCAN clustering. This repository contains a sequential as well as a distributed memory implementation for the same.

> A. Sarma et al., "μDBSCAN: An Exact Scalable DBSCAN Algorithm for Big Data Exploiting Spatial Locality," 2019 IEEE International Conference on Cluster Computing (CLUSTER), Albuquerque, NM, USA, 2019, pp. 1-11, doi: 10.1109/CLUSTER.2019.8891020

## Compile

```bash
make clean
make
```

## What this repository contains

- The original μDBSCAN implementation (sequential and MPI distributed).
- Utilities to build and run the algorithm on local or multi-node MPI setups.

## Adaptations in this fork / local copy

This workspace contains a few practical adaptations that make the code easier to run on single shared datasets and to profile:

- Single-file CSV input support: The original code expected a custom format (first line: number of points, second line: dimensionality, then whitespace-separated coordinates). `fileReadSingle` has been extended to also accept plain CSV files where each non-empty line is a point (comma- or whitespace-separated values). This makes it easy to run on standard datasets without reformatting.
- MPI single-file distribution: When running with multiple MPI ranks, if the provided input path is a regular file, rank 0 will read the entire file and distribute points across ranks (balanced) using MPI_Scatterv. This keeps filesystem access to a single process and simplifies running with `mpirun -np >1` on a shared filesystem.
- Per-rank files preserved: If the input path is a directory, the original behaviour is preserved (each rank opens `<inputDir>/out_<rank>`).
- Timing logs: Added detailed MPI-timed debug logs for the file-read/distribute stage, partitioning stage, and the extra-points exchange phase. These print to stderr (rank 0) and help profile the runtime.

## How to run

- Build the project:

```bash
make clean
make
```

- Run sequentially (no mpirun):

```bash
# sequential binary is `output` (it will run single-process if mpirun not used)
./output <input.csv> <epsilon> <minpts> <MinDeg> <MaxDeg> <outprefix>
```

- Run with MPI and a single shared CSV file (recommended for quick testing):

```bash
mpirun -np 4 ./output some_file.csv 0.1 10 10 20 out
```

This will have rank 0 read `some_file.csv` and distribute points to the other ranks. Timing logs will be printed to stderr from rank 0.

- Run with pre-split per-rank files (original behavior):

```bash
# directory must contain files named out_0, out_1, ...
mpirun -np 4 ./output <inputDir> 0.1 10 10 20 out
```

## Notes and next steps

- The CSV reader is permissive: it accepts comma-separated lines or whitespace-separated values. Lines starting with `#` or blank lines are ignored.
- If you want strict CSV parsing (quoting, escaped commas), we can swap the ad-hoc parser for a small CSV library; happy to add that if you prefer.

If you'd like, I can also add a small test dataset and a short script that runs a couple of benchmark cases and summarizes the timing logs.
