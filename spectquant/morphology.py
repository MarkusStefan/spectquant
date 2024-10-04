"""
UTILITY COLLECTION OF FUNCTIONS FOR MORPHOLOGICAL OPERATIONS
"""

import math
try:
    from typing import Union, Optional, Tuple
except (ImportError, ModuleNotFoundError):
    from collections.abc import Union, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.signal import fftconvolve

import nibabel as nib

try:
    # import fails if cuda is not available
    import cupy as cp
    from cupyx.scipy.ndimage import binary_erosion as cp_binary_erosion
    from cupyx.scipy.ndimage import binary_dilation as cp_binary_dilation
    from cupyx.scipy.signal import fftconvolve as cp_fftconvolve
except (ImportError, ModuleNotFoundError):
    cp = np
    cp_binary_erosion = binary_erosion
    cp_fftconvolve = fftconvolve
    cp_binary_dilation = binary_dilation

from spectquant import utils

def compute_suv(spect: nib.nifti1.Nifti1Image,
                seg: Optional[Union[np.ndarray,
                                    nib.nifti1.Nifti1Image]] = None,
                cube_vol: Union[int, float] = 1,
                metric: str = 'cm^3',
                method: str = 'peak',
                use_convolution: bool = True,
                use_gpu: bool = False,
                round_up: bool = False) -> Union[
                    Tuple[Tuple[Tuple[int, int], Tuple[int, int],
                                Tuple[int, int]], float],
                    Tuple[Tuple[int, int, int], float],
                    float]:
    """
    Locate and compute the SUV_peak in a 3D SPECT scan.

    Args:
        spect: SPECT scan in the form of a nifti image.
        seg: Optional binary segmentation to narrow down the search space.
        cube_vol: Volume of interest (VOI).
        metric: Volume metric.
        method: Whether to compute SUV_{peak, max, mean}.
        round_up: Whether to round up or down when translating VOI to nr of voxels.

    Returns:
        method = peak:  Tuple containing the coordinates regions of the KxKxK cube and the SUV_peak.
        method = max: Tuple containing the coordinates of the maximum voxel and the SUV_max.
        method = mean: SUV_mean.
    """
    mtx = spect.get_fdata()
    shape = mtx.shape

    if seg is not None:
        if seg.shape != shape and isinstance(seg, nib.nifti1.Nifti1Image):
            seg_ = utils.resample_img(seg, spect)
        elif seg.shape != shape and isinstance(seg, np.ndarray):
            raise ValueError(f"Segmentation must be either provided as a Nifti\
                              image for resamplint or of same shape as SPECT image:\n\
                              Segmentation: {seg.shape}\nSPECT: {shape}!")

        if isinstance(seg_, nib.nifti1.Nifti1Image):
            seg_mask: np.ndarray = seg_.get_fdata()
        elif isinstance(seg_, np.ndarray):
            seg_mask = seg_
        else:
            raise ValueError("Segmentation must be either provided as a Nifti\
                              image for resampling or of same shape as SPECT image.")

        if seg_mask.shape != shape:
            raise ValueError(
                f"Segmentation must be of same shape as SPECT image:\n\
                    Segmentation: {seg_mask.shape}\nSPECT: {shape}")

        mtx = mtx * seg_mask  # multiply SPECT by segmentation mask

    voxel_sizes = np.sqrt(np.sum(spect.affine**2, axis=0))[:3]
    mm3 = voxel_sizes[0] * voxel_sizes[1] * \
        voxel_sizes[2]  # volume of a voxel in mm3

    if metric == 'cm^3' or metric.lower() == 'ml':
        cube_vol = cube_vol * 1000  # convert to mm3 if cm3
    elif metric == 'mm^3':
        pass  # already in mm3

    cm3 = cube_vol / mm3
    if round_up:
        K = math.floor(cm3 ** (1 / 3))
    else:
        # compute K nr of voxels for each side of the cube
        K = math.ceil(cm3 ** (1 / 3))

    # if the shape is at least K*K*K
    if shape[0] < K or shape[1] < K or shape[2] < K:
        raise ValueError(
            f"The input SPECT scan must have dimensions of at least {K}x{K}x{K}.")

    if method == 'max':
        max_value = np.max(mtx)
        max_index = np.unravel_index(np.argmax(mtx), mtx.shape)
        return max_value, max_index

    if method == 'mean':
        # only compute mean of non_zero voxels (only over the segmentation
        # mask)
        non_zero_voxels = mtx[mtx != 0]
        mean_value = np.mean(non_zero_voxels)
        mean_index = ''  # nothing meaningful to return
        return mean_value, mean_index

    # find the indices of the non-zero voxels
    non_zero_indices = np.nonzero(mtx)

    # init variables to default values
    min_i, max_i = 0, 0
    min_j, max_j = 0, 0
    min_k, max_k = 0, 0

    # escape block if an error occurs -> brute force through entire image
    if len(non_zero_indices) == 0:
        print("non_zero_indices is empty. Restricting search space was not possible -> \
              now brute-forcing through the entire image.")
    else:
        try:
            # find the minimum and maximum indices along each axis
            # this constructs a "box" around the segmentation
            # this reduces down the search space
            min_i, max_i = np.min(
                non_zero_indices[0]), np.max(
                non_zero_indices[0])
            min_j, max_j = np.min(
                non_zero_indices[1]), np.max(
                non_zero_indices[1])
            min_k, max_k = np.min(
                non_zero_indices[2]), np.max(
                non_zero_indices[2])
        except IndexError as e:
            print(
                f"IndexError occurred:\t{e}\nRestricting search space was not possible ->\
                    now brute-forcing through the entire image.")
        except ValueError as e:
            print(
                f"ValueError occurred:\t{e}\nRestricting search space was not possible ->\
                    now brute-forcing through the entire image.")
        except TypeError as e:
            print(
                f"TypeError occurred:\t{e}\nRestricting search space was not possible ->\
                    now brute-forcing through the entire image.")

    # init variables to track the max sum and its coordinates
    max_sum = 0
    max_coords = None

    if use_convolution and use_gpu:
        peak_val, max_coords = _convolve_on_cuda(mtx, K)
        return peak_val, max_coords

    if use_convolution and not use_gpu:
        peak_val, max_coords = _convolve_on_cpu(mtx, K)
        return peak_val, max_coords

    # do not use convolution ... brute force through the entire image
    # init over bounding box of the non-zero volume to find the K*K*K cube
    # with the highest sum
    for i in range(max(0, min_i - 1), min(shape[0] - 2, max_i + 1)):
        for j in range(max(0, min_j - 1), min(shape[1] - 2, max_j + 1)):
            for k in range(max(0, min_k - 1), min(shape[2] - 2, max_k + 1)):
                # extract the current K*K*K cube
                current_cube = mtx[i:i + K, j:j + K, k:k + K]
                # compute the sum of the current cube
                current_sum = np.sum(current_cube)
                # update max_sum & max_coords if the current sum is greater
                if current_sum > max_sum:
                    max_sum = current_sum
                    max_coords = ((i, i + K), (j, j + K), (k, k + K))

    # return tuple of highes possible arithmetic mean (peak) and the
    # coordinates
    return max_sum / (K * K * K), max_coords


def _convolve_on_cuda(input_mtx, kernel_size: int):
    # use cuda supported cupy
    mtx = cp.array(input_mtx)
    kernel = cp.ones((kernel_size, kernel_size, kernel_size), dtype=mtx.dtype)

    # Compute the sum of every KxKxK sub-cube using convolution
    # for very large arrays, consider using fftconvolve instead for efficiency
    # cube_sums = convolve(mtx, kernel, mode='constant', cval=0)
    cube_sums = cp_fftconvolve(mtx, kernel, mode='same')
    # The result is an array where each element represents the sum of a KxKxK cube
    # originating from the corresponding element in the original array
    # Find the maximum sum and its index
    max_sum = cp.max(cube_sums)
    max_pos = cp.unravel_index(cp.argmax(cube_sums), cube_sums.shape)
    max_coords = ((max_pos[0], max_pos[0] + kernel_size), (max_pos[1],
                  max_pos[1] + kernel_size), (max_pos[2], max_pos[2] + kernel_size))
    return max_sum / (kernel_size**3), max_coords


def _convolve_on_cpu(input_mtx, kernel_size: int):
    # use regular numpy
    kernel = np.ones((kernel_size, kernel_size, kernel_size),
                     dtype=input_mtx.dtype)

    # Compute the sum of every KxKxK sub-cube using convolution
    # For very large arrays, consider using fftconvolve instead for efficiency
    # cube_sums = convolve(mtx, kernel, mode='constant', cval=0)
    cube_sums = fftconvolve(input_mtx, kernel, mode='same')
    # The result is an array where each element represents the sum of a KxKxK cube
    # originating from the corresponding element in the original array
    # Find the maximum sum and its index
    max_sum = np.max(cube_sums)
    max_pos = np.unravel_index(np.argmax(cube_sums), cube_sums.shape)
    max_coords = ((max_pos[0], max_pos[0] + kernel_size), (max_pos[1],
                  max_pos[1] + kernel_size), (max_pos[2], max_pos[2] + kernel_size))
    return max_sum / (kernel_size**3), max_coords


def erode_segmentation(seg: nib.nifti1.Nifti1Image,
                       mm_to_erode: Union[int, float],
                       use_gpu: bool = False,
                       vizz_struct: bool = False,
                       vizz_type: str = 'static') -> Union[nib.nifti1.Nifti1Image, None]:
    """
    Removes mmm_to_erode from segmentation surface to control for false positives.
    Args:
        seg: segmentation as nifti image.
        mmm_to_erode: mm to remove. The passed in number will be translated to
            voxel sizes (rounded down).
        vizz_struct: optionally, the structuring element can be visualized.
    Returns:
        vizz_struct=False: eroded image.
        vizz_struct=True: None.

    """

    # extract data & voxel sizes from  Nifti seg
    data = seg.get_fdata()
    if use_gpu:
        pp = cp
        data = cp.array(data)
        # voxel_sizes = pp.sqrt(pp.sum(cp.array(seg.affine)**2, axis=0))[:3]
    else:
        pp = np

    voxel_sizes = np.sqrt(np.sum(seg.affine**2, axis=0))[:3]

    # compute number of voxels to erode based on mm_to_erode
    voxels_to_erode = [int(mm_to_erode / voxel_size)
                       for voxel_size in voxel_sizes]

    # build custom erosion structuring element
    struct = pp.ones(voxels_to_erode)
    x = struct.shape[0] - 1
    y = struct.shape[1] - 1
    z = struct.shape[2] - 1
    # creates a "+"-shaped structure element
    struct[0, 0, 0] = 0
    struct[x, 0, 0] = 0
    struct[0, y, 0] = 0
    struct[0, 0, z] = 0
    struct[x, y, 0] = 0
    struct[0, y, z] = 0
    struct[x, 0, z] = 0
    struct[x, y, z] = 0

    if vizz_struct:
        if vizz_type == 'static':
            if use_gpu:
                utils.static_vol_vizz(struct.get())
            else:
                utils.static_vol_vizz(struct)
        else:
            if use_gpu:
                utils.vol_vizz(struct.get(), opacity=.9)
            else:
                utils.vol_vizz(struct, opacity=.9)
            return None

    # erode segmentation using binary erosion
    if use_gpu:
        # uses logical and operation, .get() to convert back to numpy
        eroded_data = cp_binary_erosion(data, structure=struct).get()
    else:
        # uses logical and operation
        eroded_data = binary_erosion(data, structure=struct)

    # establish new Nifti seg with the eroded data
    eroded_seg = nib.Nifti1Image(eroded_data, seg.affine, seg.header)

    return eroded_seg


def dilate_segmentation(seg: nib.nifti1.Nifti1Image,
                        mm_to_dilate: Union[int, float],
                        use_gpu: bool = False,
                        vizz_struct: bool = False,
                        vizz_type: str = 'static') -> Union[nib.nifti1.Nifti1Image, None]:
    """
    Expands segmentation surface according to mm_to_dilate.
    Args:
        seg: segmentation as nifti image.
        mm_to_dilate: mm to add. The passed in number will be translated to voxel \
            sizes (rounded down).
        use_gpu: whether to use GPU for computation.
        vizz_struct: optionally, the structuring element can be visualized.
        vizz_type: 'static' or 'dynamic' visualization.
    Returns:
        vizz_struct=False: dilated image.
        vizz_struct=True: None.
    """

    # extract data & voxel sizes from  Nifti seg
    data = seg.get_fdata()
    if use_gpu:
        pp = cp
        data = cp.array(data)
        # voxel_sizes = pp.sqrt(pp.sum(cp.array(seg.affine)**2, axis=0))[:3]
    else:
        pp = np

    voxel_sizes = np.sqrt(np.sum(seg.affine**2, axis=0))[:3]

    # compute number of voxels to erode based on mm_to_erode
    voxels_to_erode = [int(mm_to_dilate / voxel_size)
                       for voxel_size in voxel_sizes]

    # build custom erosion structuring element
    struct = pp.ones(voxels_to_erode)
    x = struct.shape[0] - 1
    y = struct.shape[1] - 1
    z = struct.shape[2] - 1
    # creates a "+"-shaped structure element
    struct[0, 0, 0] = 0
    struct[x, 0, 0] = 0
    struct[0, y, 0] = 0
    struct[0, 0, z] = 0
    struct[x, y, 0] = 0
    struct[0, y, z] = 0
    struct[x, 0, z] = 0
    struct[x, y, z] = 0

    if vizz_struct:
        if vizz_type == 'static':
            if use_gpu:
                utils.static_vol_vizz(struct.get())
            else:
                utils.static_vol_vizz(struct)
        else:
            if use_gpu:
                utils.vol_vizz(struct.get(), opacity=.9)
            else:
                utils.vol_vizz(struct, opacity=.9)
            return None

    # dilate segmentation using binary dilation
    if use_gpu:
        # uses logical and operation, .get() to convert back to numpy
        dilated_data = cp_binary_dilation(data, structure=struct).get()
    else:
        # uses logical and operation
        dilated_data = binary_dilation(data, structure=struct)

    # establish new Nifti seg with the eroded data
    dilated_seg = nib.Nifti1Image(dilated_data, seg.affine, seg.header)

    return dilated_seg
