# 3DGS, why not pinhole camera model?

## Removal of NDC-space calculation from 3DGS custom CUDA Kernel

Force diff-gaussian-rasterization-pinhole reinstall:

```shell
conda env create -f environment.yml
conda activate myenv
pip install ./submodules/diff-gaussian-rasterization --force-reinstall 
pip install ./submodules/diff-gaussian-rasterization-pinhole --force-reinstall 
pip install ./submodules/simple-knn --force-reinstall 
pip install ./submodules/fused-ssim --force-reinstall 

# Be aware if using nvcc 12.5
# First use pythorch 12.6 to ensure compilation of above CUDA kernels:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# But for trainig go for updated pytorch version
pip3 uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Check import:

```shell
python -c "from diff_gaussian_rasterization_pinhole import GaussianRasterizationSettings, GaussianRasterizer"
```

Test 1 - Almost same loss than Vanilla-3DGS without densification:

```shell
python train.py -s ../data/custom/aparecida_8 --model_path out_aparecida_8_pinhole --iterations 100 --densify_from_iter 99 --disable_viewer --use_pinhole
    
    ...
    p_orig: 0.9583, 1.4465, 4.2750
    p_view: -0.1199, -0.6288, 3.9625
    point_image: 244.4541, 87.2647

    p_orig: 0.9572, 1.4454, 4.2739
    p_view: 0.0667, 0.3103, 4.6715
    point_image: 259.9760, 167.0363
    ...

    [ITER 100] Evaluating train: L1 0.17146 PSNR 13.54199 LPIPS 0.11557 SSIM 0.11000 Gaussians 790 LOSS 0.31516


python train.py -s ../data/custom/aparecida_8 --model_path out_aparecida_8_vanilla --iterations 100 --densify_from_iter 99 --disable_viewer

    ...
    p_orig: 0.9583, 1.4465, 4.2750
    p_view: -0.1199, -0.6288, 3.9625
    point_image: 243.9541, 86.7647

    p_orig: 0.9572, 1.4454, 4.2739
    p_view: 0.0667, 0.3103, 4.6715
    point_image: 259.4759, 166.5363
    ...

    [ITER 100] Evaluating train: L1 0.17133 PSNR 13.55009 LPIPS 0.11559 SSIM 0.11002 Gaussians 790 LOSS 0.31506

 ```

Test 2 - Compare vanilla-3DGS when doing some densifications:

```shell
python train.py -s ../data/custom/aparecida_8 --model_path out_aparecida_8_vanilla --iterations 500 --densify_from_iter 99 --disable_viewer

    [ITER 500] Evaluating train: L1 0.07042 PSNR 20.41661 LPIPS 0.08268 SSIM 0.14240 Gaussians 4979 LOSS 0.22786 EFFICIENCY 0.051812

python train.py -s ../data/custom/aparecida_8 --model_path out_aparecida_8_pinhole --iterations 500 --densify_from_iter 99 --disable_viewer --use_pinhole

    [ITER 500] Evaluating train: L1 0.07469 PSNR 19.39686 LPIPS 0.08820 SSIM 0.13632 Gaussians 1859 LOSS 0.23249 EFFICIENCY 0.063366

python train.py -s ../data/custom/aparecida_8 --model_path out_aparecida_8_pinhole_best --iterations 500 --densify_from_iter 99 --disable_viewer --use_pinhole --densify_grad_threshold 0.000005

    [ITER 500] Evaluating train: L1 0.06560 PSNR 20.61450 LPIPS 0.07955 SSIM 0.14370 Gaussians 10615 LOSS 0.22374 EFFICIENCY 0.036244

```

 Check results:

```cmd
..\..\viewers\bin\SIBR_gaussianViewer_app.exe -m .\out_aparecida_8_pinhole --iteration 500
..\..\viewers\bin\SIBR_gaussianViewer_app.exe -m .\out_aparecida_8_vanilla --iteration 500
```