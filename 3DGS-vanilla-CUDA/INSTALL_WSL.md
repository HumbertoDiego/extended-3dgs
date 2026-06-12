## New GS for Windows Subsystem for Linux (WSL) requirements - Commit 54c035f (13/03/2025):
- PS> wsl --install -d Ubuntu
- To solve unmet dependency for libtinfo5 (only Ubuntu 24.04):

    - sudo nano /etc/apt/sources.list.d/ubuntu.sources # add

            Types: deb
            URIs: http://old-releases.ubuntu.com/ubuntu/
            Suites: lunar
            Components: universe
            Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

    - sudo apt update

- CUDA TOOLKIT 12.4: 

    - wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
    - sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
    - wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-wsl-ubuntu-12-4-local_12.4.0-1_amd64.deb
    - sudo dpkg -i cuda-repo-wsl-ubuntu-12-4-local_12.4.0-1_amd64.deb
    - sudo cp /var/cuda-repo-wsl-ubuntu-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
    - sudo apt-get update
    - sudo apt-get -y install cuda-toolkit-12-4 nvidia-cuda-toolkit
    - nvcc --version # Build cuda_12.0.r12.0/compiler.32267302_0

- Anaconda:

    - sudo apt update
    - wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
    - chmod +x Anaconda3-2024.10-1-Linux-x86_64.sh
    - ./Anaconda3-2024.10-1-Linux-x86_64.sh

- git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
- cd gaussian-splatting 
- git checkout 54c035f7834b564019656c3e3fcc3646292f727d
- conda create --name gs python=3.9
- conda activate gs
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 # --> this requires python 3.9+
- pip install tqdm plyfile opencv-python joblib
- python -c "import torch;print(torch.cuda.is_available())" # True
- BUG correction:

    - ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_PREFIX}/lib/libstdc++.so
    - ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_PREFIX}/lib/libstdc++.so.6

for diff_gaussian_rasterization:

- pip wheel ./submodules/diff-gaussian-rasterization
- pip install ./diff_gaussian_rasterization-0.0.0-cp39-cp39-linux_x86_64.whl

for simple-knn:

- pip wheel ./submodules/simple-knn
- pip install ./simple_knn-0.0.0-cp39-cp39-linux_x86_64.whl

for fused-ssim:

- pip wheel ./submodules/fused-ssim
- pip install ./fused_ssim-0.0.0-cp39-cp39-linux_x86_64.whl


- System python:

    - sudo apt install software-properties-common
    - sudo add-apt-repository ppa:deadsnakes/ppa
    - sudo apt update
    - sudo apt install python3.9 python3.12 python3.14 python-is-python3
    - sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
    - sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.12 2
    - sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.14 3
    - sudo update-alternatives --config python # escolher python3.9
    

