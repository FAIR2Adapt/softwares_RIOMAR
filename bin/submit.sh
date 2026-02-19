#!/bin/bash
#PBS -q gpu
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=16:ngpus=2:mem=512gb

cd /home1/datawork/todaka/git/softwares_RIOMAR/bin

source "$HOME/.bashrc"
micromamba activate /home1/datawork/todaka/conda-env/riomar

ENV=/home1/datawork/todaka/conda-env/riomar
export PROJ_LIB="$ENV/share/proj"
export GDAL_DATA="$ENV/share/gdal"
export PYTHONUNBUFFERED=1

python -c "import os; print('PROJ_LIB=', os.environ.get('PROJ_LIB')); import pyproj; print('pyproj data dir ok')"

~
echo "Host: $(hostname)"
echo "Python: $(which python)"
#python --version
#which python

echo "Python: $(which python)"
python --version

# --- run ---
#python Create_healpix_from_nc.py > run_nc.log 2>&1
echo "python Create_healpix_processes_by_year.py  2>&1"
#python Create_healpix_processes_by_year.py  > run_process_all.log 2>&1
ipython -c "%run Create_healpix_processes_by_year.py"  > run_process_zchunk.log 2>&1
#python Create_healpix.py > run.log 2>&1
#python Create_healpix_processes_patched.py > run_patch.log 2>&1
~
