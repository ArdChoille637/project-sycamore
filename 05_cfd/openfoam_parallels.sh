#!/usr/bin/env bash
# OpenFOAM setup in a Parallels Linux VM for Project Sycamore CFD.
#
# Purpose: run this script INSIDE a fresh Ubuntu 22.04 VM in Parallels
#          to get OpenFOAM v2412 + SU2 7.x ready for samara CFD cases.
#
# Steps to get here:
#   1. Parallels Desktop → File → New → "Download Ubuntu 22.04"
#   2. Allocate ≥4 CPU cores, ≥8 GB RAM, ≥40 GB disk
#   3. Boot VM, open Terminal inside VM, paste or run this script:
#        bash openfoam_parallels.sh
#   4. After reboot (or new terminal): source /opt/openfoam/etc/bashrc
#      Then: foamVersion  (should print openfoam12 or similar)
#
# Caveats:
#   - Install takes ~10-20 min (downloads ~3 GB).
#   - Parallels shared folders are at /media/psf/Home on the VM.
#   - macOS native OpenFOAM via Homebrew is possible but limited on arm64;
#     the VM path is more reliable for production runs.

set -e

echo "=== Step 1: system deps ==="
sudo apt-get update -q
sudo apt-get install -y -q \
    apt-transport-https ca-certificates curl gnupg \
    build-essential cmake gfortran git \
    python3 python3-pip python3-numpy \
    libboost-dev libfftw3-dev liblapack-dev libblas-dev \
    paraview  # post-processor

echo "=== Step 2: OpenFOAM v12 from openfoam.org ==="
curl -s https://dl.openfoam.org/gpg.key | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/openfoam.gpg
sudo sh -c 'echo "deb https://dl.openfoam.org/ubuntu focal main" > /etc/apt/sources.list.d/openfoam.list'
sudo apt-get update -q
sudo apt-get install -y openfoam12

echo "=== Step 3: SU2 7.x ==="
pip3 install --user su2  2>/dev/null || true
# If pip version is too old, build from source:
#   git clone https://github.com/su2code/SU2.git && cd SU2
#   ./meson.py build -Denable-pywrapper=true
#   cd build && ninja install

echo "=== Step 4: source OpenFOAM in .bashrc ==="
BASHRC="$HOME/.bashrc"
LINE='source /opt/openfoam12/etc/bashrc'
grep -qF "$LINE" "$BASHRC" || echo "$LINE" >> "$BASHRC"

echo ""
echo "=== Done. Open a new terminal and run: foamVersion ==="
echo ""
echo "Recommended first Sycamore CFD case:"
echo "  cp -r \$FOAM_TUTORIALS/incompressible/simpleFoam/pitzDaily sycamore_airfoil"
echo "  # Replace blockMeshDict with samara geometry (export from FreeCAD or Blender)"
echo "  blockMesh && simpleFoam"
echo ""
echo "Samara geometry notes:"
echo "  - Wing profile: NACA 0012 thin cambered plate (first pass)"
echo "  - Span: 0.30 m, chord at 70% span: 0.05 m"
echo "  - Simulate rotating reference frame (MRF) at Omega=50 rad/s"
echo "  - Freestream: Vd = 3 m/s downward (autorotation descent)"
echo "  - Extract: Cl, Cd per span station → validate BEM coefficients"
