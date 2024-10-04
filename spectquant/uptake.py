"""
Class for automatic computation of heart myocardial uptake volume from SPECT images.
"""
try:
    from typing import Optional, Dict, List, Union
except ImportError:
    from collections.abc import Optional, Dict, List, Union

import os
import gc
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib

from spectquant import utils
from spectquant import morphology


class UptakeVol:

    """
    Class for automatic computation of radioactive tracer uptake volume from SPECT images.

    Args:
        spect: dictionary of Nifti1Image objects with patient id and SPECT images
        spect_path: path to SPECT images directory
        segs: dictionary of Nifti1Image objects with patient id and segmentations
        segs_path: path to segmentations directory
        segs_subset: list of names of the segmentations to consider for the 
            uptake volume computation.
            Select from ['heart_myocardium', 'heart_atrium_left', 'heart_ventricle_left', 
            'heart_atrium_right', 'heart_ventricle_right', 'aorta', 'pulmonary_artery']
        mm_to_dilate: number of mm to dilate the segmentation
        approach: approach to compute the uptake volume. 
            Options are 'threshold' and 'threshold-bb' (bounding-box).
        threshold: if an integer is provided, it will be used as the threshold to binarize 
            the SPECT image. If a float is provided, it will be used as the percentile of the 
            maximum value to threshold the SPECT image.
    """

    def __init__(self,
                 spect: Optional[Dict[str, nib.nifti1.Nifti1Image]] = None,
                 spect_path: str = None,
                 segs: Optional[Dict[str, nib.nifti1.Nifti1Image]] = None,
                 segs_path: str = None,
                 segs_subset: List[str] = None,
                 mm_to_dilate: Union[int, float] = 10,
                 approach: str = 'threshold',
                 threshold: Union[int, float] = None) -> None:

        self.spect = spect
        self.spect_path = spect_path
        self.segs = segs
        self.segs_path = segs_path
        self.segs_subset = segs_subset
        self.mm_to_dilate = mm_to_dilate
        self.approach = approach
        self.threshold = threshold

        if segs_subset is None:
            self.segs_subset = ['heart_myocardium']

        if self.spect is None and self.spect_path is None:
            raise ValueError(
                "Either SPECT images or path to SPECT images must be provided")
        if self.spect is None and self.spect_path is not None:
            self.spect = nib.load(self.spect_path)
        if self.spect is not None:
            self.affine = self.spect.affine

        if self.segs is None and self.segs_path is None:
            raise ValueError(
                "Either segmentations or path to segmentations must be provided")
        if self.segs is None and self.segs_path is not None:
            self.segs = {
                f.split('.')[0]: nib.load(os.path.join(self.segs_path, f))
                for f in os.listdir(self.segs_path)
                if f.endswith('.nii.gz') and f.split('.')[0] in self.segs_subset
            }

        self.voxel_vol = np.prod(self.spect.header.get_zooms())
        self.threshold_mask = None
        self.uptake = None
        self.BB = None  # bounding box (union of all segmentations)
        self.effective_threshold = None
        self.final_mask = None
        self.uptake_nifti = None

    def _free_memory(self, verbose: bool = False) -> None:
        """
        Free memory by deleting ALL class variables.
        """
        for class_var in self.__dict__:
            print(f"Deleting {class_var}...") if verbose else None
            del class_var
        gc.collect()

    def compute_uptake_vol(self,
                           compute_number: bool = True,
                           verbose: bool = False) -> float | nib.nifti1.Nifti1Image:
        """
        Compute the myocardial uptake volume.

        Args:
            compute_number: If False, the volume to capture will be returned. 
                If True, the volume in mm^3 will be returned.
            verbose: whether to execute print statements
        """

        if self.approach in ('threshold', 'threshold-bb'):
            if self.threshold is None or self.threshold == 0:
                self.effective_threshold = 0.0
            elif isinstance(self.threshold, int) or self.threshold >= 1:
                self.effective_threshold = self.threshold
            elif isinstance(self.threshold, float) and self.threshold < 1:
                maximum = np.max(self.spect.get_fdata())
                self.effective_threshold = maximum * self.threshold
            else:
                raise ValueError(
                    'Threshold must be an integer >1 or a float in the interval (0,1]')

            self.threshold_mask = self.spect.get_fdata() > self.effective_threshold

            assert len(
                self.segs_subset) > 0, "No segmentations chosen - now computing 'threshold' approach"
            if len(self.segs_subset) == 0:
                self.approach = 'threshold'
                warnings.warn(
                    "No segmentations chosen due to empty 'segs_subset' - now computing 'threshold' approach"
                )

            if self.approach == 'threshold-bb':
                # dilate segmentation masks
                print("Dilating segmentations...") if verbose else None
                bbs = {}
                for segmentation in self.segs_subset:
                    bb = morphology.dilate_segmentation(
                        self.segs[segmentation],
                        mm_to_dilate=self.mm_to_dilate,
                        use_gpu=True
                    )
                    bbs[segmentation] = bb

                # merge (get the set union) the dilated segmentations (aka bounding boxes)
                print("Merging bounding boxes...") if verbose else None
                # remove first item from bbs dictionary
                first_seg = bbs.pop(self.segs_subset[0])
                BB = first_seg
                for bb in bbs.values():
                    BB = self._get_set_plus(BB, bb, compute_number=False)
                self.BB = BB
                if self.BB.shape != self.spect.get_fdata().shape:
                    print(
                        "Resampling threshold mask to SPECT image shape") if verbose else None
                    self.spect = utils.resample_img(
                        img=self.spect, resample_to_img=self.BB)
                    nifti_threshold_mask = nib.nifti1.Nifti1Image(
                        self.threshold_mask.astype(int), self.affine
                    )
                    self.threshold_mask = utils.resample_img(img=nifti_threshold_mask,
                                                             resample_to_img=self.BB).get_fdata()

                self.final_mask = self.threshold_mask * self.BB.get_fdata()

            elif self.approach == 'threshold':
                self.final_mask = self.threshold_mask

            else:
                raise ValueError(
                    "Invalid approach. Choose from ['threshold', 'threshold-bb']")

            self.uptake = self.spect.get_fdata() * self.final_mask

            # ensure thresholding has worked:
            # assign 0 where self.uptake <= self.effective_threshold
            self.uptake[self.uptake <= self.effective_threshold] = 0

            assert self.final_mask is not None, "self.final_mask is None"
            assert self.spect.get_fdata() is not None, "self.spect.get_fdata() is None"
            assert self.spect.get_fdata().shape == self.final_mask.shape and \
                self.uptake.shape == self.final_mask.shape, "Shapes of SPECT image and final mask do not match"

            if self.uptake is None:
                raise ValueError("self.uptake is None")
            if self.voxel_vol is None:
                raise ValueError("self.voxel_vol is None")

            if compute_number:
                return np.count_nonzero(self.uptake) * self.voxel_vol
            self.uptake_nifti = nib.nifti1.Nifti1Image(
                self.uptake, self.affine)
            return self.uptake_nifti

    @staticmethod
    def _get_set_plus(setA: np.ndarray | nib.nifti1.Nifti1Image,
                      setB: np.ndarray | nib.nifti1.Nifti1Image,
                      compute_number: bool = True) -> np.ndarray | nib.nifti1.Nifti1Image:
        """
        Get the set union of voxels of two segmentations
        """
        affine = None
        if isinstance(setA, nib.nifti1.Nifti1Image):
            setA_ = setA.get_fdata()
            affine = setA.affine
        else:
            setA_ = setA
        if isinstance(setB, nib.nifti1.Nifti1Image):
            setB_ = setB.get_fdata()
            affine = setB.affine
        else:
            setB_ = setB

        set_plus = np.logical_or(setA_, setB_)

        if compute_number:
            return np.count_nonzero(set_plus)

        if isinstance(affine, np.ndarray):
            return nib.nifti1.Nifti1Image(set_plus.astype(int), affine)
        return set_plus

    def plot_heatmaps(self):
        """
        Plot heatmaps of the sum along each axis of the original SPECT image and 
        the extracted uptake volume.
        """
        original_data: np.ndarray = self.spect.get_fdata()
        extracted_roi: np.ndarray = self.uptake

        _, axes = plt.subplots(2, 3, figsize=(18, 12))

        sns.heatmap(np.sum(original_data, axis=0),
                    ax=axes[0, 0], cmap='viridis')
        axes[0, 0].set_title('Heatmap: Sum along x-axis')

        sns.heatmap(np.sum(original_data, axis=1),
                    ax=axes[0, 1], cmap='viridis')
        axes[0, 1].set_title('Heatmap: Sum along y-axis')

        sns.heatmap(np.sum(original_data, axis=2),
                    ax=axes[0, 2], cmap='viridis')

        axes[0, 2].set_title('Heatmap: Sum along z-axis')

        sns.heatmap(np.sum(extracted_roi, axis=0),
                    ax=axes[1, 0], cmap='viridis')
        axes[1, 0].set_title('Heatmap: Sum along x-axis')

        sns.heatmap(np.sum(extracted_roi, axis=1),
                    ax=axes[1, 1], cmap='viridis')
        axes[1, 1].set_title('Heatmap: Sum along y-axis')

        sns.heatmap(np.sum(extracted_roi, axis=2),
                    ax=axes[1, 2], cmap='viridis')
        axes[1, 2].set_title('Heatmap: Sum along z-axis')

        plt.show()

    def vizz_uptake(self) -> None:
        """
        Visualize the uptake volume
        """
        if self.uptake is None:
            raise ValueError(
                "Septum segmentation not computed yet - run .compute_septum_volume()")
        utils.seg_vizz(imgs=[self.uptake],
                       names=[
                           f'Uptake Volume of {" - ".join(self.segs_subset)}']
                       )
