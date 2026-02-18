#!/bin/bash
#PBS -q gpu
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=16:ngpus=2:mem=512gb

# --- micromamba environment ---
# If micromamba isn't already on PATH, load it here (site-specific).
# Example: module load micromamba

# Initialize micromamba shell support
#eval "$(micromamba shell hook -s bash)"

# Activate your environment by absolute path
#micromamba activate /home1/datawork/todaka/conda-env/riomar
cd /home1/datawork/todaka/git/softwares_RIOMAR/bin
ENV=/home1/datawork/todaka/conda-env/riomar
export PATH="$ENV/bin:$PATH"

echo "Host: $(hostname)"
echo "Python: $(which python)"
#python --version
#which python

echo "Python: $(which python)"
python --version

# --- run ---
#python Create_healpix_processes.py > run.log 2>&1
#python Create_healpix.py > run.log 2>&1
python Create_healpix_processes_patched.py > run.log 2>&1
~
