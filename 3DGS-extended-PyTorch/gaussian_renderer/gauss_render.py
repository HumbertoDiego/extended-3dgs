#
# Edited from diff-gaussian-rasterization\diff_gaussian_rasterization\__init__.py
#

from typing import NamedTuple
import torch.nn as nn
import torch
import math
from utils.graphics_utils import fov2focal
from utils.weight_utils import curv3d, curv2d, diskindex, smallsizeindex, reusageindex
#
# Taken from torch-splatting\gaussian_splatting\utils\sh_utils.py
#

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

#
# Taken from torch-splatting\gaussian_splatting\gauss_render.py
#

def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance
    # symm = strip_symmetric(actual_covariance)
    # return symm

def build_covariance_2d(
    mean3d, cov3d, viewmatrix, tan_fovx, tan_fovy, focal_x, focal_y, low_pass_filter=0.3
):
    # The following models the steps outlined by equations 29
	# and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	# Additionally considers aspect / scaling of viewport.
	# Transposes used to account for row-/column-major conventions.
    t = (mean3d @ viewmatrix[:3,:3]) + viewmatrix[-1:,:3]

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    # Eq.29 locally affine transform 
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros((mean3d.shape[0], 3, 3), device="cuda").to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    # J[..., 2, 0] = tx / t.norm(dim=-1) # discard
    # J[..., 2, 1] = ty / t.norm(dim=-1) # discard
    # J[..., 2, 2] = tz / t.norm(dim=-1) # discard
    W = viewmatrix[:3,:3].T # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0,2,1)
    
    # add low pass filter here according to E.q. 32
    filter = torch.eye(2,2, device="cuda").to(cov2d) * low_pass_filter
    return cov2d[:, :2, :2] + filter[None]

def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points) # object space
    points_h = points_o @ projmatrix # projmatrix here == full_proj_transform
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_frustum = p_view[..., 2] >= 0.2
    return p_proj, p_view, in_frustum

@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    my_radius = (3.0 * torch.sqrt(torch.max(lambda1, lambda2))).ceil().to(torch.int32)
    return my_radius

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    # Clip the rectangles to the image dimensions
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    thin_width = (rect_max[..., 0] - rect_min[..., 0]) == 0
    thin_height = (rect_max[..., 1] - rect_min[..., 1]) == 0
    jump_narrow_gaussian = thin_width | thin_height
    return (rect_min, rect_max), ~jump_narrow_gaussian

def build_color(active_sh_degree, means3D, shs, camera_center):
        rays_o = camera_center
        rays_d = means3D - rays_o
        color = eval_sh(active_sh_degree, shs.permute(0,2,1), rays_d)
        color = (color + 0.5).clip(min=0.0)
        return color
 
class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool
    camera_center: torch.Tensor
    image_name :str
    tile_size: int
    use_techniques: tuple
    low_pass_filter: float
    min_weight: float
    max_weight: float
    progress: float
    k: int

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        visible = positions[..., 2] >= 0.2
        return visible
 
    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings
        use_curvature3d, use_curvature2d, use_diskindex, use_fixed_to_small_index , use_smallsizeindex, use_fixed_big_small, use_fixedindex, only_evaluate_weights  = raster_settings.use_techniques

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Do not Invoke C++/CUDA rasterization routine...
        # Do every thing here:  
        mean_ndc, mean_view, in_frustum = projection_ndc(means3D, 
                                            viewmatrix=raster_settings.viewmatrix, 
                                            projmatrix=raster_settings.projmatrix)
        mean_ndc.requires_grad_(True)
        mean_ndc.retain_grad()
        
        # mean_ndc = mean_ndc[in_frustum]   # This change the shape of everything
        # mean_view = mean_view[in_frustum] # This change the shape of everything
        # Instead, mask frustum while tile rendering below
        depths = mean_view[:,2]

        color = build_color(active_sh_degree=raster_settings.sh_degree, 
                            means3D=means3D, 
                            shs=shs, 
                            camera_center=raster_settings.camera_center)

        cov3d = build_covariance_3d(scales, rotations)

        fy = raster_settings.image_height / (2.0 * raster_settings.tanfovy)
        fx = raster_settings.image_width / (2.0 * raster_settings.tanfovx)

        cov2d = build_covariance_2d(
                mean3d=means3D, 
                cov3d=cov3d, 
                viewmatrix=raster_settings.viewmatrix,
                tan_fovx=raster_settings.tanfovx, 
                tan_fovy=raster_settings.tanfovy, 
                focal_x=fx, 
                focal_y=fy,
                low_pass_filter=raster_settings.low_pass_filter)

        mean_coord_x = ((mean_ndc[..., 0] + 1) * raster_settings.image_width - 1.0) * 0.5
        mean_coord_y = ((mean_ndc[..., 1] + 1) * raster_settings.image_height - 1.0) * 0.5
        means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)

        pix_coord = torch.stack(torch.meshgrid(torch.arange(raster_settings.image_width), 
                                            torch.arange(raster_settings.image_height), 
                                            indexing='xy'), dim=-1).to('cuda')
        
        # Não posso = 0 em cov2d original pois haverá sua inversa mais a frente
        # Instead, set 0 in radius itself
        radii = get_radius(cov2d)
        radii[~in_frustum] = 0  # set 0 when not in frustum

        # cada gaussiana projetada com seus means2D e raio serão representados 
        # por retângulos para comparar com o tile mais à frente:
        rect, size_ok = get_rect(means2D, radii, width=raster_settings.image_width, height=raster_settings.image_height)
        radii[~size_ok] = 0  # set 0 when gaussian is too thin
        
        # Tile rendering
        render_color = torch.ones(*pix_coord.shape[:2], 3).to('cuda')
        render_depth = torch.zeros(*pix_coord.shape[:2], 1).to('cuda')
        render_alpha = torch.zeros(*pix_coord.shape[:2], 1).to('cuda')



        full_weight = torch.full_like(opacities, float('nan'))
        # default TILE_SIZE=16 --> lower memory consumption!
        # Increase to speed up, but increase memory consumption.
        TILE_SIZE = raster_settings.tile_size 
        pattern = torch.linspace(1.0, 0.1, steps=10)
        for h in range(0, raster_settings.image_height, TILE_SIZE):
            for w in range(0, raster_settings.image_width, TILE_SIZE):
                # check if the rectangle penetrate the tile
                over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h) # mins = top,left
                over_br = rect[1][..., 0].clip(max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1) # maxs = bottom,right
                in_tile = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
                # 3D gaussian in the tile, in front of camera, and not too thin
                in_mask = in_tile & in_frustum & size_ok
                if not in_mask.sum() > 0:
                    continue
                mask_idx = in_mask.nonzero(as_tuple=True)[0]


                P = in_mask.sum()
                right_limit = raster_settings.image_width if w+TILE_SIZE>raster_settings.image_width else w+TILE_SIZE
                bottom_limit = raster_settings.image_height if h+TILE_SIZE>raster_settings.image_height else h+TILE_SIZE

                tile_coord = pix_coord[h:bottom_limit, w:right_limit].flatten(0,-2)
                sorted_depths, index = torch.sort(depths[in_mask])
                sorted_means2D = means2D[in_mask][index]
                sorted_cov2d = cov2d[in_mask][index] # P 2 2
                sorted_conic = sorted_cov2d.inverse() # inverse of variance
                
                sorted_opacity = opacities[in_mask][index]
                sorted_color = color[in_mask][index]
                dx = (sorted_means2D[None,:] - tile_coord[:,None,:]) # B P 2

                #### Added in 07-06-2025
                sorted_scales = scales[in_mask][index]
                if use_curvature3d:
                    weights = curv3d(sorted_scales, raster_settings.max_weight, raster_settings.min_weight)
                elif use_curvature2d:
                    weights = curv2d(sorted_cov2d, raster_settings.max_weight, raster_settings.min_weight)
                elif use_diskindex: # Added in 18-07-2025
                    weights = diskindex(sorted_scales, raster_settings.max_weight, raster_settings.min_weight)
                elif use_smallsizeindex: # Added in 22-07-2025
                    weights = smallsizeindex(sorted_scales, raster_settings.max_weight, raster_settings.min_weight)
                elif use_fixed_big_small: # Added in 06-08-2025
                    # progress = 0 -> favorece bigscale index
                    # progress = 1 -> favorece smallscale index
                    if raster_settings.k:
                        progress = 1 - 2.718281**(-1*raster_settings.k*raster_settings.progress) # exp
                    else:
                        progress = raster_settings.progress # linear
                    S = smallsizeindex(sorted_scales, raster_settings.max_weight, raster_settings.min_weight)
                    weights = (1-progress) + progress * S
                elif use_fixed_to_small_index: # Added in 27-09-2025
                    # progress = 0 -> favorece fixed index
                    # progress = 1 -> favorece smallscale index
                    if raster_settings.k:
                        progress = 1 - 2.718281**(-1*raster_settings.k*raster_settings.progress) # exp
                    else:
                        progress = raster_settings.progress # linear
                    S = smallsizeindex(sorted_scales, raster_settings.max_weight, raster_settings.min_weight)
                    fixed_weights = raster_settings.max_weight * torch.ones_like(sorted_scales[:,0])
                    weights = (1-progress) * fixed_weights + progress * S
                elif use_fixedindex: # Added in 10-08-2025
                    weights = raster_settings.max_weight * torch.ones_like(sorted_scales[:,0])
                    # weights = pattern.repeat(scales[:,0].shape[0] // 10 + 1)[:scales[:,0].shape[0]].cuda()
                    # weights = weights[in_mask][index]
                else:
                    weights = torch.ones_like(sorted_scales[:,0])

                #### Added in 22-09-2025
                if use_fixed_to_small_index:
                    full_weight[mask_idx[index]] = S[..., None]
                else:
                    full_weight[mask_idx[index]] = weights[..., None]
                # full_weight[in_mask][index] = weights[...,None] # pytoch não consegue atribuir valores em uma indexação encadeada  
                
                #### Added in 07-06-2025
                if only_evaluate_weights:
                    weights = torch.ones_like(sorted_scales[:,0])
                    
                gauss_weight = weights[None] * torch.exp(-0.5 * ( dx[:, :, 0]**2 * sorted_conic[:, 0, 0] 
                                                        + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
                                                        + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]
                                                        + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]))
                #### Added in 07-06-2025
                alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) # B P 1
                # if (alpha < 1.0f / 255.0f) continue; #L407 
                # more than 50% of alphas falls here in early stages of training:
                # alpha[alpha < 1.0/255.0] = 0
                T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)
                acc_alpha = (alpha * T).sum(dim=1)
                tile_color = (T * alpha * sorted_color[None]).sum(dim=1) + (1- acc_alpha) * raster_settings.bg
                tile_depth = ((T * alpha) * 1/ sorted_depths[None,:,None]).sum(dim=1) # invdepths
                render_color[h:bottom_limit, w:right_limit] = tile_color.reshape(bottom_limit-h, right_limit-w, -1)
                render_depth[h:bottom_limit, w:right_limit] = tile_depth.reshape(bottom_limit-h, right_limit-w, -1)
                render_alpha[h:bottom_limit, w:right_limit] = acc_alpha.reshape(bottom_limit-h, right_limit-w, -1)
        
        # rets = {
        #     "alpha": render_alpha,
        # }
        color = render_color.permute(2,0,1)
        invdepths = render_depth.permute(2,0,1)
        
        return color, radii, invdepths, mean_ndc, full_weight

