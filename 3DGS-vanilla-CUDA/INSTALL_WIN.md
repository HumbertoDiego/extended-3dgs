## New GS for Windows 11 requirements - Commit 54c035f (13/03/2025):
- git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
- cd gaussian-splatting 
- git checkout 54c035f7834b564019656c3e3fcc3646292f727d

- Anaconda:
    - conda create --name gs python=3.9
    - conda activate gs
    - [CUDA TOOLKIT 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)
    - nvcc --version # Build cuda_12.4.r12.4/compiler.34097967_0
    - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 # --> this requires python 3.9+
    - pip install tqdm plyfile opencv-python joblib
    - Microsoft Visual C++ 14.0 or greater:
        - Download [Visual Studio](https://visualstudio.microsoft.com/pt-br/)
        - VisualStudioSetup.exe --> Visual Studio Comunity 2022 --> check "Desktop development with C++"

    for diff_gaussian_rasterization:

    - pip wheel .\submodules\diff-gaussian-rasterization\
    - pip install .\diff_gaussian_rasterization-0.0.0-cp39-cp39-win_amd64.whl

    for simple-knn:

    - pip wheel .\submodules\simple-knn
    - pip install .\simple_knn-0.0.0-cp39-cp39-win_amd64.whl

    for fused-ssim:

    - pip wheel .\submodules\fused-ssim\
    - pip install .\simple_knn-0.0.0-cp39-cp39-win_amd64.whl

- pixi com Python3.9:
    ```
    [workspace]
    authors = ["HumbertoDiego <humberto-xingu@live.com>"]
    channels = ["conda-forge", "https://conda.anaconda.org/nvidia", "pytorch"]
    name = "extended-3dgs"
    platforms = ["win-64"]
    version = "0.1.0"

    [tasks]

    [dependencies]
    python = "==3.9.23"
    cuda-version = "12.4.*"
    cuda-toolkit = "==12.4.0"
    pytorch-cuda = ">=12.4,<13"
    pytorch = "==2.4.1"
    numpy = ">=2.0.2,<3"
    tqdm = ">=4.67.1,<5"
    plyfile = ">=1.1,<2"
    joblib = ">=1.5.1,<2"
    opencv = ">=4.12.0,<5"
    wheel = ">=0.45.1,<0.46"
    pip = ">=25.2,<26"
    pillow = ">=10.4.0,<11"
    ```
    - pixi shell
    - cd 3DGS-vanilla-pinhole

    for diff_gaussian_rasterization:

    - pip wheel .\submodules\diff-gaussian-rasterization\
    - pip install .\diff_gaussian_rasterization-0.0.0-cp39-cp39-win_amd64.whl

    for simple-knn:

    - pip wheel .\submodules\simple-knn
    - pip install .\simple_knn-0.0.0-cp39-cp39-win_amd64.whl

    for fused-ssim:

    - pip wheel .\submodules\fused-ssim\
    - pip install .\simple_knn-0.0.0-cp39-cp39-win_amd64.whl

- pixi
    - pixi init
	- pixi add python=3.12
	- pixi add pip ipykernel jupyter
	- pixi shell
    - pip install torch --index-url https://download.pytorch.org/whl/cu124 # esse comando mistura com o python3.12 do SO -> ModuleNotFoundError: No module named 'torch'
    - pixi add imageio matplotlib ipywidgets ipython-autotime plyfile tqdm plyfile joblib open3d mlflow 
	- pixi add opencv
	- cd 3DGS-pinhole
	- pip wheel .\submodules\diff-gaussian-rasterization\ --no-build-isolation; pip install .\diff_gaussian_rasterization-0.0.0-cp312-cp312-win_amd64.whl
	- pip wheel .\submodules\simple-knn\ --no-build-isolation; pip install .\simple_knn-0.0.0-cp312-cp312-win_amd64.whl
	- pip wheel .\submodules\fused-ssim\ --no-build-isolation; pip install .\fused_ssim-0.0.0-cp312-cp312-win_amd64.whl
	- pip wheel .\submodules\diff-gaussian-rasterization-pinhole-v2 --no-build-isolation; pip install .\diff_gaussian_rasterization_pinhole_v2-0.0.0-cp312-cp312-win_amd64.whl --force-reinstall


- for SIBR_viewers:

    - Download Windows Precompiled at [SIBR_core](https://repo-sam.inria.fr/fungraph/sibr-release/sibr-core/install.zip) place at `.\GaussianViewTest\viewers` 
    - cd .\GaussianViewTest\viewers\bin
    - .\SIBR_gaussianViewer_app.exe -m ../../model
    - Train something and open the viewer to check on-the-fly results:
        - .\SIBR_remoteGaussian_app 
        - Open another shell
        - python train.py -s .\GaussianViewTest\train\
