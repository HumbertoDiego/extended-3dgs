# HumbertoDiego - extended-3dgs

- Original 3DGS readme: [README_vanilla.md](https://github.com/HumbertoDiego/extended-3dgs/blob/main/3DGS-vanilla-CUDA/README_vanilla.md)
- Original 3DGS install instructions for Windows 11: [INSTALL_WIN.md](https://github.com/HumbertoDiego/extended-3dgs/blob/main/3DGS-vanilla-CUDA/INSTALL_WIN.md)
- Original 3DGS install instructions for WSL-Windows 11: [INSTALL_WSL.md](https://github.com/HumbertoDiego/extended-3dgs/blob/main/3DGS-vanilla-CUDA/INSTALL_WSL.md)
- Details about PyTorch implementation of `foward` method: [README.md](https://github.com/HumbertoDiego/extended-3dgs/blob/main/3DGS-vanilla-PyTorch/README.md)
- Details about performance improved PyTorch implementation of `foward` method: [README.md](https://github.com/HumbertoDiego/extended-3dgs/blob/main/3DGS-vanilla-PyTorch_AMP_COMPILED_LOADER/README.md)
- Details about PyTorch modified implementation to account for curvatures: [README.md](https://github.com/HumbertoDiego/extended-3dgs/blob/main/3DGS-extended-PyTorch/README.md)

Overall Requirements:
- CUDA nvcc, no caso 12.4
- nvcc --version # ... Cuda compilation tools, release 12.4, V12.4.131
- Anaconda
    - conda create -n gs python=3.12 pip ipykernel jupyter
    - conda activate gs
    - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    - pip install accelerate einops imageio matplotlib ipywidgets ipython-autotime plyfile opencv-python open3d tensorboard
- pixi
	- pixi init
	- pixi add python=3.12
	- pixi shell
    - pixi add pip ipykernel jupyter imageio matplotlib ipywidgets ipython-autotime plyfile tqdm plyfile joblib open3d mlflow opencv numpy scikit-learn
	- pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 # esse comando mistura com o python3.12 do SO -> ModuleNotFoundError: No module named 'torch'
	- cd 3DGS-pinhole
	- pip wheel .\submodules\diff-gaussian-rasterization\ --no-build-isolation
	- pip install .\diff_gaussian_rasterization-0.0.0-cp312-cp312-win_amd64.whl
	- pip wheel .\submodules\simple-knn\ --no-build-isolation
	- pip install .\simple_knn-0.0.0-cp312-cp312-win_amd64.whl
	- pip wheel .\submodules\fused-ssim\ --no-build-isolation
	- pip install .\fused_ssim-0.0.0-cp312-cp312-win_amd64.whl
	- pip wheel .\submodules\diff-gaussian-rasterization-pinhole --no-build-isolation
	- pip install .\diff_gaussian_rasterization_pinhole-0.0.0-cp312-cp312-win_amd64.whl --force-reinstall

Usage:
1) 3DGS original (custom CUDA kernels):
	- (gs) PS > cd 3DGS-vanilla-CUDA
	- (gs) PS > python .\train.py -s ..\data\custom\aparecida --model_path output 
	```
	Training progress:  23% 7000/30000 [03:59<19:52, 19.28it/s, Loss=0.0152967, Depth Loss=0.0000000] 
	[ITER 7000] Evaluating train: L1 0.009004752896726132 PSNR 35.76117630004883 [01/07 15:17:28]

	Training progress: 100% 30000/30000 [26:32<00:00, 18.84it/s, Loss=0.0066586, Depth Loss=0.0000000]
	[ITER 30000] Evaluating train: L1 0.004444140149280429 PSNR 43.185105133056645 [01/07 15:40:01]
	```

2) 3DGS modified with custom CUDA kernels: `foward` modified to account curvatures, but failed to implement `backward`. See [Formula_Trail.md](https://github.com/HumbertoDiego/extended-3dgs/blob/main/tests/Formula_Trail.md) for more details.
	- (gs) PS > cd 3DGS-extended-half-CUDA 
	- (gs) PS > pip install .\submodules\diff-gaussian-rasterization --force-reinstall 
	- (gs) PS > python .\train.py -s ..\data\aparecida --model_path output 
	- Fast, but got: "# RuntimeError: CUDA error: an illegal memory access was encountered"

3) Pure Pytorch implementation (no CUDA kernel) of L1 loss of 3DGS:
	- (gs) PS > cd 3DGS-vanilla-PyTorch
	- (gs) PS > python .\train.py -s ..\data\aparecida --model_path output --densify_until_size 30000 --tile_size 64
	```
	Training progress:  23% 7000/30000 [47:19<2:51:48,  2.23it/s, Loss=0.0311256, Depth Loss=0.0000000, Number of Gaussians=35216]
	[ITER 7000] Evaluating train: L1 0.01649679895490408 PSNR 30.744768142700195 [09/07 01:38:59]

	Training progress: 100% 30000/30000 [3:47:10<00:00,  2.20it/s, Loss=0.0192517, Depth Loss=0.0000000, Number of Gaussians=35216]
	[ITER 30000] Evaluating train: L1 0.012148732505738736 PSNR 33.63533058166504 [09/07 04:38:50]
	```

4) Pure Pytorch implementation of L1 loss of modified 3DGS to account for curvatures:
	- (gs) PS > cd 3DGS-extended-PyTorch
	- (gs) PS > python .\train.py -s ..\data\aparecida --model_path output --densify_until_size 30000 --tile_size 64
	- TODO: test

5) Performance improved Pytorch implementation of L1 loss of 3DGS:
	- (gs) PS > cd 3DGS-vanilla-PyTorch_AMP_COMPILED_LOADER
	- (gs) PS > python .\train.py -s ..\data\aparecida --model_path output

6) Accelerated by MultiGPU/MuliNode Pytorch implementation of L1 loss of 3DGS:
	- (gs) PS > cd 3DGS-vanilla-PyTorch_Accelerate
	- (gs) PS > accelerate launch --multi_gpu --num_processes=2 .\train.py -s ..\data\aparecida --model_path output --densify_until_size 30000 --tile_size 64

7) 3DGS with no NDC spac (custom CUDA kernel):
	- (gs) PS > cd 3DGS-pinhole;  pixi shell
	- (gs) PS > mlflow server --port 8080
	- (gs) PS > python train.py -s ../data/custom/aparecida_8 --model_path out_aparecida_8_vanilla --iterations 500 --densify_from_iter 99 --disable_viewer --experiment test2
	```
	Reading camera 27/27 [03/06 13:50:07]
	Number of points at initialisation :  790 [03/06 13:50:07]
	[ITER 100] Evaluating train: L1 0.15340 PSNR 14.35653 LPIPS 0.10443 SSIM 0.14348 Gaussians 790 LOSS 0.29403 EFFICIENCY 1000.000000 [03/06 13:50:11]
	[ITER 200] Evaluating train: L1 0.11847 PSNR 16.67281 LPIPS 0.10133 SSIM 0.15178 Gaussians 1212 LOSS 0.26442 EFFICIENCY 0.188508 [03/06 13:52:01]
	[ITER 300] Evaluating train: L1 0.10306 PSNR 17.61028 LPIPS 0.09718 SSIM 0.15435 Gaussians 1819 LOSS 0.25158 EFFICIENCY 0.096975 [03/06 13:52:05]
	[ITER 400] Evaluating train: L1 0.07893 PSNR 19.55189 LPIPS 0.09473 SSIM 0.15615 Gaussians 2964 LOSS 0.23191 EFFICIENCY 0.124186 [03/06 13:52:08]
	Training progress: 100%|████| 500/500 [02:01<00:00,  4.13it/s, Loss=0.1059364, Depth Loss=0.0000000, Number of Gaussians=4844]

	[ITER 500] Evaluating train: L1 0.06094 PSNR 21.33279 LPIPS 0.09206 SSIM 0.16194 Gaussians 4844 LOSS 0.21636 EFFICIENCY 0.105694 [03/06 13:52:11]

	[ITER 500] Saving Gaussians [03/06 13:52:11]
	🏃 View run out_aparecida_8_vanilla at: http://localhost:8080/#/experiments/822789449854788682/runs/c93c9836ce6c4bca826dfae1d32ddb95 [03/06 13:52:11]
	🧪 View experiment at: http://localhost:8080/#/experiments/822789449854788682 [03/06 13:52:11]

	Training complete. [03/06 13:52:11]
    ```
	- python train.py -s ../data/custom/aparecida_8 --model_path out_aparecida_8_pinhole --iterations 500 --densify_from_iter 99 --disable_viewer --use_pinhole --experiment test2

Notebooks inside `/tests`:
- [Gradientes_Explanation.ipynb](https://github.com/HumbertoDiego/extended-3dgs/blob/main/tests/gradientes_explanation.ipynb)
- [Raio_Explanation.ipynb](https://github.com/HumbertoDiego/extended-3dgs/blob/main/tests/raio_explanation.ipynb)
- [Colmap_Model_tests.ipynb](https://github.com/HumbertoDiego/extended-3dgs/blob/main/tests/Colmap_Model_tests.ipynb)
- [Projections_Models_tests.ipynb](https://github.com/HumbertoDiego/extended-3dgs/blob/main/tests/Projections_Models_tests.ipynb)
- [Torch_Backpropagation_tests.ipynb](https://github.com/HumbertoDiego/extended-3dgs/blob/main/tests/Torch_Backpropagation_tests.ipynb)

<!-- 
git init
git remote add 3dgs https://github.com/HumbertoDiego/extended-3dgs
git pull 3dgs main
# Do and push changes:
git add * ; git commit -m "run final"; git push -u 3dgs main
#Pull changes
git pull origin main 
 -->
