# F2A Riomar Singularity Images

This repository contains a layered stack of Singularity images for the **Fair2Adapt Riomar** project, designed for geospatial data processing with Zarr, xarray, and JupyterHub compatibility on HPC infrastructure (Ifr and others).

## 🖼️ Image Stack Overview

| Layer | Definition File | Base Image | Purpose |
|-------|----------------|------------|---------|
| **A** | `imgsingular_A_hardened_base.def` | `dhi.io/debian-base:trixie` | Hardened Debian base with essential tools |
| **B** | `imgsingular_B_conda_f2a_riomar_p311.def` | `imgsingular_A_hardened_base.sif` | Miniconda + Python 3.11 + Riomar scientific stack |
| **C** | `imgsingular_C_f2a_jupyterhub_ifr.def` | `imgsingular_B_conda_f2a_riomar_p311.sif` | Full JupyterHub/BatchSpawner for IFR HPC (and others) |

## 🛠️ Build Instructions

### Prerequisites
```bash
# Install SingularityCE (Apptainer compatible)
sudo apt install singularity-ce

# Ensure write permissions in working directory
mkdir -p ~/singularity_images
cd ~/singularity_images
```

### Sequential build

```bash
# 1. Build base hardened image (~5min)
sudo singularity build imgsingular_A_hardened_base.sif imgsingular_A_hardened_base.def

# 2. Build conda + scientific stack (~15min)
sudo singularity build imgsingular_B_conda_f2a_riomar_p311.sif imgsingular_B_conda_f2a_riomar_p311.def

# 3. Build JupyterHub image (~10min)
sudo singularity build imgsingular_C_f2a_jupyterhub_ifr.sif imgsingular_C_f2a_jupyterhub_ifr.def
```

## 🚀 Usage Examples
### Interactive Shell (Any Layer)

```bash
singularity shell imgsingular_C_f2a_jupyterhub_ifr.sif
# Inside: conda list  # env_f2a_riomar is active
```

### Jupyter Lab (Production Layer C)
TODO : provide real example
```bash
singularity run --cleanenv -B /ifs/work/ifremer:/data imgsingular_C_f2a_jupyterhub_ifr.sif
# Opens JupyterLab on port 8888
```

### Batch Job (IFR HPC Compatible)
TODO : provide real example
```bash
#!/bin/bash
#$ -cwd -l mem=16G -pe smp 4
singularity exec imgsingular_C_f2a_jupyterhub_ifr.sif python /data/your_script.py
```

## ✨ Key Features

### Layer A (Base)
- Hardened Debian Trixie from DHI
- `curl`, `wget`, `bzip2` pre-installed
- **Alternative**: Replace `From: dhi.io/debian-base:trixie` with `ubuntu:24.04` for non-hardened builds

### Layer B (Scientific Stack)
- **Miniconda 3** with `conda-forge` only (strict priority)
- **Python 3.11** environment `env_f2a_riomar` (**not 3.13+ due to Jupyter `pipes` module removal**)
- **Core libraries**: `xarray`, `zarr`, `cdshealpix`, `pystac-client`, `cartopy`, `folium`
- **Auto-activated** on container start

### Layer C (JupyterHub)
- **IFR HPC compatible (and others)**: `jupyterhub==2.3.1`, `jupyterlab==2.3.1`, `batchspawner`
- Registered kernel: `Python 3.11 (env_f2a_riomar)`
- Default `jupyter lab --ip=0.0.0.0 --allow-root`

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Auth issues with Docker Hardened images | Rebuild Layer A with `From: ubuntu:24.04` |
| `pipes` ModuleNotFoundError | Use Python 3.11 (already enforced) |
| Slow conda install | First build takes 20-25min, subsequent rebuilds use layer cache |
| Jupyter XSRF errors | Layer C disables check by default |

## ⚙️ Customization

Add packages in `%post` sections:
```bash
# Layer B: conda run -n env_f2a_riomar conda install -y new_package
# Layer C: conda run -n env_f2a_riomar pip install additional_package
```
