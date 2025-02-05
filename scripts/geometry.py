'''
Created by Basile Van Hoorick for GCD, 2024.
Fast, optimized code for "rendering" merged point clouds into pseudo-ground truth video pairs.
'''

# Library imports.
import functools
import numpy as np
import skimage
import skimage.metrics
from einops import rearrange

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False, threshold=1000)


def cartesian_from_spherical(spherical, deg2rad=False):
    '''
    :param spherical: (..., 3) array of float.
    :return cartesian: (..., 3) array of float.
    '''
    azimuth = spherical[..., 0]
    elevation = spherical[..., 1]
    radius = spherical[..., 2]
    if deg2rad:
        azimuth = np.deg2rad(azimuth)
        elevation = np.deg2rad(elevation)
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)
    cartesian = np.stack([x, y, z], axis=-1)
    return cartesian


def spherical_from_cartesian(cartesian, rad2deg=False):
    '''
    :param cartesian: (..., 3) array of float.
    :return spherical: (..., 3) array of float.
    '''
    x = cartesian[..., 0]
    y = cartesian[..., 1]
    z = cartesian[..., 2]
    radius = np.linalg.norm(cartesian, ord=2, axis=-1)
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.linalg.norm(cartesian[..., 0:2], ord=2, axis=-1))
    if rad2deg:
        azimuth = np.rad2deg(azimuth)
        elevation = np.rad2deg(elevation)
    spherical = np.stack([azimuth, elevation, radius], axis=-1)
    return spherical


def interpolate_spherical(cart_start, cart_end, alpha):
    '''
    :param cart_start: (3) array of float.
    :param cart_end: (3) array of float.
    :param alpha: single float.
    :return cart_interp: (3) array of float.
    '''
    spher_start = spherical_from_cartesian(cart_start)
    spher_end = spherical_from_cartesian(cart_end)
    if spher_end[0] - spher_start[0] > np.pi:
        spher_end[0] -= 2 * np.pi
    if spher_end[0] - spher_start[0] < -np.pi:
        spher_end[0] += 2 * np.pi
    if spher_end[1] - spher_start[1] > np.pi:
        spher_end[1] -= 2 * np.pi
    if spher_end[1] - spher_start[1] < -np.pi:
        spher_end[1] += 2 * np.pi
    spher_interp = spher_start * (1 - alpha) + spher_end * alpha
    cart_interp = cartesian_from_spherical(spher_interp)
    return cart_interp


def extrinsics_from_look_at(camera_position, camera_look_at):
    '''
    :param camera_position: (3) array of float.
    :param camera_look_at: (3) array of float.
    :return RT: (4, 4) array of float.
    '''
    # NOTE: In my convention (including Kubric and ParallelDomain),
    # the columns (= camera XYZ axes) should be: right, down, forward.

    # Calculate forward vector: Z.
    forward = (camera_look_at - camera_position)
    forward /= np.linalg.norm(forward)
    # Assume world's down vector: Y.
    world_down = np.array([0, 0, -1])
    # Calculate right vector: X = Y cross Z.
    right = np.cross(world_down, forward)
    right /= np.linalg.norm(right)
    # Calculate actual down vector: Y = Z cross X.
    down = np.cross(forward, right)

    # Construct 4x4 extrinsics matrix.
    RT = np.eye(4)
    RT[0:3, 0:3] = np.stack([right, down, forward], axis=1)
    RT[0:3, 3] = camera_position

    return RT


def camera_to_world(xyz_camera, extrinsics):
    '''
    :param xyz_camera: (..., 3) array of float.
    :param extrinsics: (4, 4) array of float.
    :return xyz_world: (..., 3) array of float.
    '''
    xyz_world = xyz_camera @ extrinsics[0:3, 0:3].T
    xyz_world += extrinsics[0:3, 3]
    return xyz_world


def world_to_camera(xyz_world, extrinsics):
    '''
    :param xyz_world: (..., 3) array of float.
    :param extrinsics: (4, 4) array of float.
    :return xyz_camera: (..., 3) array of float.
    '''
    xyz_camera = xyz_world - extrinsics[0:3, 3]
    xyz_camera = xyz_camera @ extrinsics[0:3, 0:3]
    return xyz_camera


def project_points_to_pixels(xyzrgb, K, RT, H, W, spread_radius=2):
    '''
    NOTE: Keep in mind this is much faster on GPU!
    :param xyzrgb: (N, 6) tensor of float, where N can be any value.
    :param K: (3, 3) tensor of float.
    :param RT: (4, 4) tensor of float.
    :return img_norm: (H, W, 3) tensor of float.
    :return pixel_weights: (H, W, 1) tensor of float.
    :return uv: (N, 2) tensor of float.
    :return depth: (N, 1) tensor of float.
    '''
    # NOTE: Various del statements are used to free up VRAM as soon as possible. These optimizations
    # save around 40% of VRAM usage. Also, the VRAM is proportional to the total number of workers
    # (across all processes). See also: https://pytorch.org/docs/stable/torch_cuda_memory.html

    xyzrgb = xyzrgb.type(torch.float64)
    K = K.type(torch.float64).to(xyzrgb.device)
    RT = RT.type(torch.float64).to(xyzrgb.device)

    # Extracting xyz and projecting to camera coordinates.
    xyz_world = xyzrgb[:, 0:3]  # (N, 3).
    xyz_camera = world_to_camera(xyz_world, RT)
    del xyz_world

    # Projecting to pixel coordinates.
    uv = torch.mm(K, xyz_camera.T).T  # (N, 3).
    uv = uv[:, 0:2] / uv[:, 2:3]  # Divide by z to get image coordinates.

    # Convert to integer pixel coordinates and apply mask.
    uv_int = (uv + 0.5).type(torch.int32)  # (N, 2) with (horizontal, vertical) coordinates.
    depth = xyz_camera[:, 2:3]  # Depth is z in camera coordinates.
    mask = (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W) & \
           (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H) & \
           (depth[:, 0] > 0.1)
    del xyz_camera

    # Filter points that are inside the image and have valid depth.
    rgb_filter = xyzrgb[mask][:, 3:6]  # (M, 3).
    uv_int_filter = uv_int[mask]  # (M, 2).
    depth_filter = depth[mask]  # (M, 1).
    del mask

    # Convert 2D indices to 1D. These coordinates are horizontal minor and vertical major.
    inds_flat = uv_int_filter[:, 1] * W + uv_int_filter[:, 0]
    del uv_int_filter

    # Make sure closer points are considered significantly more important in this aggregation.
    if depth_filter.max() >= 64.0:
        # We're probably dealing with ParallelDomain which has very far away points.
        strength = 256.0
        depth_filter = torch.sqrt(depth_filter)
        depth_filter = torch.clamp(depth_filter, 0.0, 32.0)
    else:
        # We're probably dealing with Kubric.
        strength = 512.0
        # NOTE: In Kubric, strength 128 or above creates overflow in float32,
        # and 1024 or above creates overflow in float64.

    depth_norm = depth_filter / depth_filter.max() * 2.0 - 1.0  # (M, 1) with values in [-1, 1].
    del depth_filter
    point_weights = torch.exp(-depth_norm * strength)  # (M, 1) and decreasing with depth.
    del depth_norm
    weighted_rgb = rgb_filter * point_weights  # (M, 3) and gets darker with depth.
    del rgb_filter

    # Normalize by accumulating weighted point-to-pixel counts (becomes denominator).
    pixel_weights_flat = torch.zeros(H * W, 1, dtype=torch.float64, device=xyzrgb.device)
    spreaded_index_add(pixel_weights_flat, inds_flat, point_weights, H, W, spread_radius)
    del point_weights
    # (H * W * 1).

    # Accumulate weighted color pixel values themselves (becomes numerator).
    img_flat = torch.zeros(H * W, 3, dtype=torch.float64, device=xyzrgb.device)
    spreaded_index_add(img_flat, inds_flat, weighted_rgb, H, W, spread_radius)
    del weighted_rgb
    # (H * W * 3).

    # Avoid division by zero, but make it clear exactly where no point contributed to a pixel.
    pixel_weights = pixel_weights_flat.view(H, W, 1)  # Reshape to (H, W, 1).
    pixel_weights[pixel_weights <= 0.0] = -1.0

    # Also calculate direct counts for debugging.
    if 0:
        count_flat = torch.zeros(H * W, 1, dtype=torch.int32, device=xyzrgb.device)  # (H * W * 1).
        ones = torch.ones_like(point_weights, dtype=torch.int32)
        count_flat.index_add_(0, inds_flat, ones)
        count = count_flat.view(H, W, 1)  # Reshape back to (H, W, 1).
        count = torch.clamp(count, min=1)  # Avoid division by zero.

    # Normalize and clip final pixel values.
    img = img_flat.view(H, W, 3)  # Reshape to (H, W, 3).
    img_norm = img / pixel_weights
    img_norm = torch.clamp(img_norm, 0.0, 1.0)
    img_norm = img_norm.type(torch.float32)

    return (img_norm, pixel_weights, uv, depth)


def spreaded_index_add(tensor, indices, values, H, W, radius):
    '''
    :param tensor: (N, C) tensor of any type.
    :param indices: (M) tensor of int with values in [0, N - 1].
    :param values: (M, C) tensor of any type.
    :param H, W, radius (int): Image dimensions and spread radius.
    :return tensor: (N, C) tensor of any type.
    '''
    # Benchmark stuff for debugging.
    if 0:
        NI = 10

        t = tensor.detach().clone().cpu()
        i = indices.detach().clone().cpu()
        v = values.detach().clone().cpu()
        for _ in range(NI):
            t.index_add_(0, i, v)
        t = tensor.detach().clone().cuda(1)
        i = indices.detach().clone().cuda(1)
        v = values.detach().clone().cuda(1)
        for _ in range(NI):
            t.index_add_(0, i, v)

    # Accumulate values at indices in-place within tensor.
    tensor.index_add_(0, indices, values)

    # NOTE: Only the above line would be the default / vanilla operation, but we wish to
    # avoid random pixel holes inbetween points, which requires a more advanced algorithm.
    left = radius // 2
    right = (radius + 1) // 2
    offset_list = []
    for dx in range(-left, right + 1):
        for dy in range(-left, right + 1):
            if dx == 0 and dy == 0:
                continue
            offset_list.append((dx, dy))
    # when radius = 1: [(1, 0), (0, 1), (1, 1)].
    # when radius = 2: [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)].
    # when radius = 3: x and y go from -1 to 2 inclusive, etc..

    for (dx, dy) in offset_list:
        # Spread values to neighboring pixels.
        inds_x = indices % W + dx
        inds_y = indices // W + dy
        shift_inds = inds_y * W + inds_x

        # Also avoid leaking across image borders.
        # shift_inds = torch.clamp(shift_inds, min=0, max=H * W - 1)
        mask = (inds_x >= 0) & (inds_x < W) & (inds_y >= 0) & (inds_y < H)
        mask_inds = shift_inds[mask]
        mask_values = values[mask] * 0.02  # Weaken as original pixels should have priority.

        tensor.index_add_(0, mask_inds, mask_values)

    return tensor


def blur_into_black(img, kernel_size=5, sigma=1.5):
    '''
    :param img: (H, W, 3) tensor of float.
    :param radius (int): Blur radius.
    '''
    black_mask = (img.sum(dim=-1) == 0.0)[None]  # (1, H, W).
    img = rearrange(img, 'h w c -> c h w')  # (3, H, W).

    # First leak valid content into invalid regions.
    img2 = gaussian_blur_masked_vectorized(img, ~black_mask, black_mask, kernel_size, sigma)

    # Then apply slight, gentle blurring to smooth the rough edges due to both spreaded_index_add
    # and previous operation.
    img2 = torchvision.transforms.functional.gaussian_blur(
        img2, kernel_size=3, sigma=0.6)

    img2 = rearrange(img2, 'c h w -> h w c')  # (H, W, 3).
    return img2


def gaussian_blur_masked_vectorized(img, borrow_mask, apply_mask, kernel_size, sigma):
    '''
    Apply Gaussian blur to an image but only considering pixels within borrow_mask for the
        convolution content, and change only pixels within apply_mask.
    :param img: (C, H, W) tensor of float.
    :param borrow_mask: (1, H, W) tensor of bool indicating which pixels to use.
    :param apply_mask: (1, H, W) tensor of bool indicating which pixels to modify.
    '''
    borrow_mask = borrow_mask.type(torch.float64)

    # in 2D:
    blur_img = torchvision.transforms.functional.gaussian_blur(
        img, kernel_size=kernel_size, sigma=sigma)
    blur_mask = torchvision.transforms.functional.gaussian_blur(
        borrow_mask, kernel_size=kernel_size, sigma=sigma)

    blur_mask = blur_mask.clamp(min=1e-7)
    leak_img = blur_img / blur_mask

    final_img = img * (~apply_mask) + leak_img * apply_mask
    return final_img

def render_from_pcl(cur_xyzrgb, spherical, K, 
                    project_size = (280, 420), render_size = (256, 384),
                    kernel_size=21, sigma=21/4.0):
    camera_pos = cartesian_from_spherical(spherical, deg2rad=True)
    camera_pos[...,2] += 1.0

    RT = torch.tensor(extrinsics_from_look_at(camera_pos, np.array([0.0, 0.0, 1.0])))
    (cur_synth1, cur_weights, cur_uv, cur_depth) = project_points_to_pixels(cur_xyzrgb, K, RT, project_size[0], project_size[1], spread_radius=1)
    blur_synth1 = blur_into_black(cur_synth1, kernel_size=kernel_size, sigma=sigma)
    blur_synth1 = F.interpolate(rearrange(blur_synth1, 'h w c -> c h w')[None], render_size, mode='bilinear', align_corners=False)[0]
    return blur_synth1.cpu()

def get_gaussian_coords(spherical):
    T = cartesian_from_spherical(spherical, deg2rad=True)
    T[...,2] += 1.0
    RT = extrinsics_from_look_at(T, np.array([0.0, 0.0, 1.0]))
    R = RT[:3,:3].T
    T = -np.matmul(R, T)
    R = R.T
    return R.astype(np.float32), T.astype(np.float32)

def construct_trajectory(spherical_start, spherical_end, trajectory, model_frames, move_time):
    '''
    :param spherical_start: (3,) array of float32.
    :param spherical_end: (3,) array of float32.
    :param trajectory (str).
    :param model_frames (int).
    :param move_time (int).
    :return spherical_src: (Tcm, 3) array of float32.
    :return spherical_dst: (Tcm, 3) array of float32.
    '''
    Tcm = model_frames

    # Determine input camera trajectory.
    spherical_src = np.tile(spherical_start[None], (Tcm, 1))
    # (Tcm, 3) array of float32.

    # Determine output camera trajectory.
    spherical_dst = np.tile(spherical_end[None], (Tcm, 1))
    if move_time >= 1:
        for t in range(0, move_time):
            if trajectory == 'interpol_linear':
                alpha = t / move_time
            elif trajectory == 'interpol_sine':
                alpha = (1.0 - np.cos(t / move_time * np.pi)) / 2.0
            else:
                raise ValueError(f'Unknown trajectory: {trajectory}')
            spherical_dst[t] = spherical_start * (1.0 - alpha) + spherical_end * alpha
    # (Tcm, 3) array of float32.

    return (spherical_src, spherical_dst)

def masked_ssim(im1, im2, mask, win_size=7, K1=0.01, K2=0.03, sigma=1.5, channel_axis=0):
    '''
    This is adapted from scikit-learn version 0.22.0 skimage.metrics.structural_similarity,
    but we allow for arbitrary non-rectangular regions to be considered only.
    NOTE: We assume these parameters:
    data_range = 1.0, full = False, gaussian_weights = False, gradient = False.
    :param im1: (C?, H, W, C?) array of float in [0, 1].
    :param im2: (C?, H, W, C?) array of float in [0, 1].
    :param mask: (H, W) array of bool.
    :return (mssim_all, mssim_mask): float x2.
    '''

    from scipy.ndimage import binary_erosion, uniform_filter
    from skimage._shared import utils
    from skimage._shared.utils import _supported_float_type, check_shape_equality, warn
    from skimage.util.arraycrop import crop

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    mask = mask.astype(bool)
    check_shape_equality(im1, im2)
    float_type = _supported_float_type(im1.dtype)

    if channel_axis is not None:
        # loop over channels
        nch = im1.shape[channel_axis]

        channel_axis = channel_axis % im1.ndim
        _at = functools.partial(utils.slice_at_axis, axis=channel_axis)

        mssims = []
        for ch in range(nch):
            ch_result = masked_ssim(
                im1[_at(ch)], im2[_at(ch)], mask, win_size=win_size, K1=K1, K2=K2, sigma=sigma,
                channel_axis=None)
            mssims.append(ch_result)

        mssims = np.mean(mssims, axis=0)
        return mssims

    use_sample_covariance = True
    ndim = im1.ndim
    filter_func = uniform_filter
    filter_args = {'size': win_size}

    # ndimage filters need floating point data
    im1 = im1.astype(float_type, copy=False)
    im2 = im2.astype(float_type, copy=False)

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(im1, **filter_args)
    uy = filter_func(im2, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(im1 * im1, **filter_args)
    uyy = filter_func(im2 * im2, **filter_args)
    uxy = filter_func(im1 * im2, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = 1.0
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # OLD:
    # compute (weighted) mean of ssim. Use float64 for accuracy.
    S_crop = crop(S, pad)
    mssim_all = np.mean(S_crop, dtype=np.float64)

    # NEW:
    mask_erode = binary_erosion(mask, iterations=pad)
    mask_crop = crop(mask_erode, pad)
    mssim_mask = np.mean(S_crop[mask_crop], dtype=np.float64)

    mssims = np.array([mssim_all, mssim_mask])
    return mssims

def calculate_metrics(gt_rgb, reproject_rgb, pred_samples):
    '''
    :param input_rgb: (Tcm, 3, Hp, Wp) array of float32 in [0, 1].
    :param gt_rgb: (Tcm, 3, Hp, Wp) array of float32 in [0, 1].
    :param reproject_rgb: (Tcm, 3, Hp, Wp) array of float32 in [0, 1].
    :param pred_samples: list of dicts with keys:
        cond_rgb, sampled_rgb, sampled_latent.
    '''
    # NOTE: This subroutine is a bit rudimentary, because it does not include baseline metrics;
    # see more advanced scripts for that.
    S = len(pred_samples)

    if S >= 1:
        pred_samples_rgb = np.stack([x for x in pred_samples], axis=0)
        # (S, Tcm, 3, Hp, Wp) array of float32 in [0, 1].
    else:
        pred_samples_rgb = []

    if reproject_rgb is not None:
        occluded_mask = (np.sum(np.abs(reproject_rgb), axis=1) <= 1e-7).astype(np.uint8)
        visible_mask = 1 - occluded_mask
        # 2x (Tcm, Hp, Wp) array of uint8 in [0, 1].

        visible_mask_bc = np.tile(visible_mask[:, None].astype(bool), (1, 3, 1, 1))
        occluded_mask_bc = np.tile(occluded_mask[:, None].astype(bool), (1, 3, 1, 1))
        # 2x (Tcm, 3, Hp, Wp) array of bool.

    frame_psnr = []
    frame_psnr_vis = []
    frame_psnr_occ = []
    frame_ssim = []
    frame_ssim_vis = []
    frame_ssim_occ = []

    for output_rgb in pred_samples_rgb:
        # output_rgb = (Tcm, 3, Hp, Wp) array of float32 in [0, 1].
        (Tcm, _, Hp, Wp) = output_rgb.shape

        cur_frame_psnr = []
        cur_frame_psnr_vis = []
        cur_frame_psnr_occ = []
        cur_frame_ssim = []
        cur_frame_ssim_vis = []
        cur_frame_ssim_occ = []

        for t in range(Tcm):

            cur_psnr = skimage.metrics.peak_signal_noise_ratio(
                output_rgb[t], gt_rgb[t], data_range=1.0)
            cur_ssim = skimage.metrics.structural_similarity(
                output_rgb[t], gt_rgb[t], data_range=1.0, channel_axis=0)

            cur_frame_psnr.append(cur_psnr)
            cur_frame_ssim.append(cur_ssim)

            if reproject_rgb is not None:
                cur_vis_mask = visible_mask_bc[t]  # (3, Hp, Wp) array of bool.
                cur_occ_mask = occluded_mask_bc[t]  # (3, Hp, Wp) array of bool.
                cur_output_vis = output_rgb[t][cur_vis_mask]
                cur_gt_vis = gt_rgb[t][cur_vis_mask]
                cur_output_occ = output_rgb[t][cur_occ_mask]
                cur_gt_occ = gt_rgb[t][cur_occ_mask]

                if cur_vis_mask.any():
                    cur_psnr_vis = skimage.metrics.peak_signal_noise_ratio(
                        cur_output_vis, cur_gt_vis, data_range=1.0)
                    cur_ssim_vis = masked_ssim(
                        output_rgb[t], gt_rgb[t], cur_vis_mask[0])[1]
                else:
                    cur_psnr_vis = np.nan
                    cur_ssim_vis = np.nan

                if cur_occ_mask.any():
                    cur_psnr_occ = skimage.metrics.peak_signal_noise_ratio(
                        cur_output_occ, cur_gt_occ, data_range=1.0)
                    cur_ssim_occ = masked_ssim(
                        output_rgb[t], gt_rgb[t], cur_occ_mask[0])[1]
                else:
                    cur_psnr_occ = np.nan
                    cur_ssim_occ = np.nan

                cur_frame_psnr_vis.append(cur_psnr_vis)
                cur_frame_ssim_vis.append(cur_ssim_vis)
                cur_frame_psnr_occ.append(cur_psnr_occ)
                cur_frame_ssim_occ.append(cur_ssim_occ)

        frame_psnr.append(cur_frame_psnr)
        frame_ssim.append(cur_frame_ssim)
        frame_psnr_vis.append(cur_frame_psnr_vis)
        frame_ssim_vis.append(cur_frame_ssim_vis)
        frame_psnr_occ.append(cur_frame_psnr_occ)
        frame_ssim_occ.append(cur_frame_ssim_occ)

    frame_psnr = np.array(frame_psnr)  # (S, Tcm) array of float.
    frame_ssim = np.array(frame_ssim)  # (S, Tcm) array of float.
    frame_psnr_vis = np.array(frame_psnr_vis)  # (S, Tcm) array of float.
    frame_ssim_vis = np.array(frame_ssim_vis)  # (S, Tcm) array of float.
    frame_psnr_occ = np.array(frame_psnr_occ)  # (S, Tcm) array of float.
    frame_ssim_occ = np.array(frame_ssim_occ)  # (S, Tcm) array of float.

    mean_psnr = np.nanmean(frame_psnr, axis=1)  # (S) array of float.
    mean_ssim = np.nanmean(frame_ssim, axis=1)  # (S) array of float.
    mean_psnr_vis = np.nanmean(frame_psnr_vis, axis=1)  # (S) array of float.
    mean_ssim_vis = np.nanmean(frame_ssim_vis, axis=1)  # (S) array of float.
    mean_psnr_occ = np.nanmean(frame_psnr_occ, axis=1)  # (S) array of float.
    mean_ssim_occ = np.nanmean(frame_ssim_occ, axis=1)  # (S) array of float.

    uncertainty = np.nanmean(np.std(pred_samples_rgb, axis=0), axis=1)
    # (Tcm, Hp, Wp) array of float32 in [0, 1].
    frame_diversity = np.nanmean(uncertainty, axis=(1, 2))
    # (Tcm) array of float32 in [0, 1].
    mean_diversity = np.nanmean(frame_diversity)
    # single float.

    if reproject_rgb is not None:
        # NOTE: To ensure correct statistics, we apply array masking instead of multiplication here.
        pred_samples_vis = [np.stack([x[t][visible_mask_bc[t]] for x in pred_samples_rgb], axis=0)
                            for t in range(Tcm)]
        pred_samples_occ = [np.stack([x[t][occluded_mask_bc[t]] for x in pred_samples_rgb], axis=0)
                            for t in range(Tcm)]
        # 2x List-T of (S, N) of float32 in [0, 1].
        frame_diversity_vis = np.array([np.nanmean(np.std(x, axis=0)) for x in pred_samples_vis])
        frame_diversity_occ = np.array([np.nanmean(np.std(x, axis=0)) for x in pred_samples_occ])
        # 2x (Tcm) array of float32 in [0, 1].
        mean_diversity_vis = np.nanmean(frame_diversity_vis)
        mean_diversity_occ = np.nanmean(frame_diversity_occ)
        # 2x single float.

    metrics_dict = dict()
    metrics_dict['frame_psnr'] = frame_psnr
    metrics_dict['frame_ssim'] = frame_ssim
    metrics_dict['frame_diversity'] = frame_diversity
    metrics_dict['mean_psnr'] = mean_psnr
    metrics_dict['mean_ssim'] = mean_ssim
    metrics_dict['mean_diversity'] = mean_diversity

    if reproject_rgb is not None:
        metrics_dict['frame_psnr_vis'] = frame_psnr_vis
        metrics_dict['frame_ssim_vis'] = frame_ssim_vis
        metrics_dict['frame_psnr_occ'] = frame_psnr_occ
        metrics_dict['frame_ssim_occ'] = frame_ssim_occ
        metrics_dict['frame_diversity_vis'] = frame_diversity_vis
        metrics_dict['frame_diversity_occ'] = frame_diversity_occ
        metrics_dict['mean_psnr_vis'] = mean_psnr_vis
        metrics_dict['mean_ssim_vis'] = mean_ssim_vis
        metrics_dict['mean_psnr_occ'] = mean_psnr_occ
        metrics_dict['mean_ssim_occ'] = mean_ssim_occ
        metrics_dict['mean_diversity_vis'] = mean_diversity_vis
        metrics_dict['mean_diversity_occ'] = mean_diversity_occ

    return (metrics_dict, uncertainty)