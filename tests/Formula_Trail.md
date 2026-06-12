
## Trilha para encontrar as implementações das fórmulas de composição da cor do spixels

Chamadas ao rasterizador começam no arquivo `train.py#L81`

```
net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer,  
				   use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
```

Proveniente de  `from gaussian_renderer import render, network_gui`

Em seguida, no arquivo `gaussian_renderer/__init__.py` dentro da função `render`:

```
rasterizer(
    means3D = means3D,
    means2D = means2D,
    dc = dc,
    shs = shs,
    colors_precomp = colors_precomp,
    opacities = opacity,
    scales = scales,
    rotations = rotations,
    cov3D_precomp = cov3D_precomp)
```

Proveniente de `rasterizer = GaussianRasterizer(raster_settings=raster_settings)`, por sua vez proveniente de `from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer`

Seguem para `diff-gaussian-rasterization/diff-gaussian-rasterization/__init__.py#L158 class GaussianRasterizer`

```
def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
 ...
        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )
```

Ou seja, seguem para `def rasterize_gaussians` que no mesmo arquivo `diff-gaussian-rasterization/diff-gaussian-rasterization/__init__.py#L21` é definido como:

```
def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )
```

Seguem para `diff-gaussian-rasterization/diff-gaussian-rasterization/__init__.py#L44 class _RasterizeGaussians`:

```
class _RasterizeGaussians:
    # ...
    # Invoke C++/CUDA rasterizer
    num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)
```

Seguem para` diff-gaussian-rasterization/ext.cpp`:

```
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
```

Seguem para `diff-gaussian-rasterization/rasterize_points.cu function RasterizeGaussiansCUDA`:

```
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug)
    //...
    rendered = CudaRasterizer::Rasterizer::forward(...)
```

Goes to `diff-gaussian-rasterization/cuda_rasterizer/rasterizer.h` static member function of the Rasterizer class within the CudaRasterizer namespace:

```
...
namespace CudaRasterizer
{
	class Rasterizer
	{
	...
    static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* depth,
			bool antialiasing,
			int* radii = nullptr,
			bool debug = false);
```

Seguem para `diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu`:

```
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* depth,
	bool antialiasing,
	int* radii,
	bool debug)
{
    ...
    CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		geomState.depths,
		depth), debug)
}
```

Seguem para `diff-gaussian-rasterization/cuda_rasterizer/foward.cu` função `render` no namespace `FOWARD`:

```
void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* depths,
	float* depth)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths, 
		depth);
}
```

Que por fim, executa `diff-gaussian-rasterization/cuda_rasterizer/foward.cu` função [renderCUDA](https://github.com/HumbertoDiego/extended-3dgs/blob/main/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu?plain=1#L276) na GPU:


```
// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth)
{
    ...
    // Eq. (3) from 3D Gaussian splatting paper.
    for (int ch = 0; ch < CHANNELS; ch++)
        C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
    ...
```

Que por sua vez, aplica a equação 3 do paper a cada iteração `i` para determinar a cor de cada pixel:

$$
C = \sum_{i=1}^{N}\alpha_i c_i \underbrace{\prod_{j=1}^{i-1} (1-\alpha_j)}_{T_i}
$$

## Estudo da inserção da curvatura no cálculo da composição da cor.

### Como calcular a curvatura:

Pela construção de $\Sigma$, seus autovalores $\lambda$ são derivados dos fatores de escala (scale.x, scale.y e scale.z) pois:

$$\Sigma = RSS^TR^T$$
$$\Sigma R = RSS^TR^TR$$
$$\Sigma R = RSS^T$$
$$R^{-1}\Sigma R = R^{-1}RSS^T$$
$$R^{-1}\Sigma R = SS^T = 
\begin{bmatrix}
\lambda_0 &             & \\
            & \lambda_1 & \\
            &             & \lambda_2
\end{bmatrix}
$$

Portanto, $\Sigma$ e $SS^T$ são matrizes similares e possuem os mesmos autovalores. Logo $S$ possui a forma:

$$
S = diag(\sqrt{\lambda_0},\sqrt{\lambda_1},\sqrt{\lambda_2})
$$

A curvatura $\sigma_{\mu}$ será então calculada por:

$$
\sigma_{\mu} = \frac{min(\lambda_0,\lambda_1,\lambda_2)}{\lambda_0+\lambda_1+\lambda_2} = \frac{min(scale_x^2,scale_y^2,scale_z^2)}{scale_x^2+scale_y^2+scale_z^2}
$$

### Alteração no código proposta:

```
    // Proposition
    float min_val = min(min(scale.x, scale.y), scale.z);
    float curvatura = min_val*min_val / (scale.x*scale.x + scale.y*scale.y + scale.z*scale.z);
```

### Simple test

Em `tests/test.cpp`:

```
#include <stdio.h>
#include <algorithm>
#include <glm/glm.hpp>
using namespace std; 

void getCurvatura(const glm::vec3 scale, float& curvatura) {
    // Proposition
    float min_val = min(min(scale.x, scale.y), scale.z);
    curvatura = min_val*min_val / (scale.x*scale.x + scale.y*scale.y + scale.z*scale.z);
}

int main() {
    printf("curvatura...\n");
    float curvatura;
    glm::vec3 scale(1.0f,2.0f,3.0f);
    getCurvatura(scale, curvatura);
    printf("%.9g",curvatura);
    return 0;
}
```
Compilar:

```
> nvcc -Isubmodules/diff-gaussian-rasterization/third_party/glm ./tests/test.cpp -o ./tests/test ; ./tests/test.exe
    test.cpp
    curvatura...
    0.0714285746
```

### Local onde existe a variável `scales` para ser adicionada à função `renderCUDA`:

Em vez de levar a variável `curvatura` à função `renderCUDA`, vamos levar a variável `scales`. Para tanto, devemos fazer as seguintes alterações, primeiramente em [forward.cu](https://github.com/HumbertoDiego/extended-3dgs/blob/main/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu):

```
//L274
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	...
	float* __restrict__ invdepth,
	const glm::vec3* scales)

//L402
void FORWARD::render(
	...
	float* depth,
	const glm::vec3* scales)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		...
		depth,
		scales);
}

```

Bem como, no arquivo de cabeçalho [foward.h](https://github.com/HumbertoDiego/extended-3dgs/blob/main/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.h):

```
//L52
void render(
		...
		float* depth,
		const glm::vec3* scales
		);
```

Em seguida onde é chamada a função `render` dentro do namespace `FOWARD` em [rasterizer_impl.cu](https://github.com/HumbertoDiego/extended-3dgs/blob/main/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu)

```
//L325
CHECK_CUDA(FORWARD::render(
		...
		depth,
		(glm::vec3*)scales), debug)

```

Neste ponto, recebemos a variável `scales` dentro da função `renderCUDA` mas ainda não a usamos. Para marcar este checkpoint vamos checar se o código original ainda funciona, podemos compilar com:

```
> pip wheel .\submodules\diff-gaussian-rasterization\
	...
	Successfully built diff_gaussian_rasterization
> pip install .\diff_gaussian_rasterization-0.0.0-cp39-cp39-win_amd64.whl --force-reinstall
	...
	Successfully installed diff-gaussian-rasterization-0.0.0
```

### Alteração da função renderCUDA

```cpp
// diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L309
	// Allocate storage for batches of collectively fetched data.
	...
	__shared__ glm::vec3 collected_scale[BLOCK_SIZE];

// diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L330
	// Collectively fetch per-Gaussian data from global to shared
	...
	collected_scale[block.thread_rank()] = scales[coll_id];

// diff-gaussian-rasterization/cuda_rasterizer/forward.cu#L356
	// Proposition
	glm::vec3 scale = collected_scale[j];
	float min_val = min(min(scale.x, scale.y), scale.z);
	float curvatura = min_val*min_val / (scale.x*scale.x + scale.y*scale.y + scale.z*scale.z);
	// End proposition
	// Eq. (2) from 3D Gaussian splatting paper.
	// Obtain alpha by multiplying with Gaussian opacity
	// and its exponential falloff from mean.
	// Avoid numerical instabilities (see paper appendix). 
	float alpha = min(0.99f, curvatura * con_o.w * exp(power)); // <-- curvatura inserted here
```

Para checar se o novo código ainda compila:

```
> pip wheel .\submodules\diff-gaussian-rasterization\
	...
	Successfully built diff_gaussian_rasterization
> pip install .\diff_gaussian_rasterization-0.0.0-cp39-cp39-win_amd64.whl --force-reinstall
	...
	Successfully installed diff-gaussian-rasterization-0.0.0
```

Para processar:

```
> python train.py -s .\GaussianViewTest\train\
	Optimizing 
	Output folder: ./output/099030d5-9 [17/03 17:13:47]
	Tensorboard not available: not logging progress [17/03 17:13:47]
	Reading camera 301/301 [17/03 17:13:48]
	Loading Training Cameras [17/03 17:13:48]
	Loading Test Cameras [17/03 17:13:53]
	Number of points at initialisation :  182686 [17/03 17:13:53]
	Training progress:   3%|███▋                                  | 800/30000 [02:04<2:03:58,  3.93it/s, Loss=0.3462326, Depth Loss=0.0000000]
```

## Trilha para encontrar as implementações das derivadas e atualização dos parâmetros das gaussianas

A atualização dos parâmetros intrínsecos de cada Gaussiana é chamado ao final de cada iteração de treino no arquivo `train.py#L142`

```
	loss.backward()
```

No PyTorch, o gradiente dos tensores inicializados com `requires_grad=True` e que acumulam gradientes, são calculados ao chamar `.backward()` com relação a algum valor escalar.

O notebook [Gradientes_Explanation.ipynb](https://github.com/HumbertoDiego/extended-3dgs/blob/main/tests/Gradientes_Explanation.ipynb) explica maiores detalhes da implementação do `.backward()` no chamado <i>custom CUDA kernel</i>.
