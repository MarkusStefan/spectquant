"""
Class for automatic computation of heart myocardial uptake volume from SPECT images.
"""
try:
    from typing import Optional, Dict, List, Union
except ImportError:
    from collections.abc import Optional, Dict, List, Union

import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib

from spectquant import utils
from spectquant import morphology

class SegmentVol:
    """
    Class for automatic computation of the volume of one or multiple segmentation masks.

    Args:
        segs: dictionary of Nifti1Image objects with patient id and segmentations
        segs_path: path to segmentations directory
        segs_subset: list of names of the segmentations to consider for the volume computation.
            Select from ['heart_myocardium', 'heart_atrium_left', 'heart_ventricle_left', 
            heart_atrium_right', 'heart_ventricle_right', 'aorta', 'pulmonary_artery'].
        mm_to_erode: number of mm to erode the segmentation
        mm_to_dilate: number of mm to dilate the segmentation
    """

    def __init__(self,
                 segs: Optional[Dict[str, nib.nifti1.Nifti1Image]] = None,
                 segs_path: str = None,
                 segs_subset: Union[List[str], str] = None,
                 mm_to_erode: Union[int, float] = None,
                 mm_to_dilate: Union[int, float] = None) -> None:

        self.segs = segs
        self.segs_path = segs_path
        self.segs_subset = [segs_subset] if isinstance(
            segs_subset, str) else segs_subset
        self.mm_to_erode = mm_to_erode
        self.mm_to_dilate = mm_to_dilate
        if self.segs_subset is None:
            self.segs_subset = ['heart_myocardium']

        if self.segs is None and self.segs_path is None:
            raise ValueError(
                "Either segmentations or path to segmentations must be provided")
        if self.segs is None and self.segs_path is not None:
            self.segs = {
                f.split('.')[0]: nib.load(os.path.join(self.segs_path, f))
                for f in os.listdir(self.segs_path)
                if f.endswith('.nii.gz') and f.split('.')[0] in self.segs_subset
            }

        if self.mm_to_erode and self.mm_to_dilate:
            raise ValueError(
                "Either mm_to_erode or mm_to_dilate can be provided, not both")

        if len(self.segs) == 0:
            raise ValueError(
                "No segmentations found in the provided directory")

        # get the voxel volume from the first segmentation
        self.voxel_vol = np.prod(list(self.segs.values())[
                                 0].header.get_zooms())

        self.BB = None  # bounding-box
        self.voxels = None

    def _free_memory(self, verbose: bool = False) -> None:
        """
        Free memory by deleting ALL class variables.
        """
        for class_var in self.__dict__.keys():
            print(f"Deleting {class_var}...") if verbose else None
            del class_var
        gc.collect()

    def compute_vol(self,
                    compute_number: bool = True,
                    verbose: bool = False) -> float | nib.nifti1.Nifti1Image:
        """
        Compute the myocardial uptake volume.

        Args:
            compute_number: If False, the volume to capture will be returned. 
                If True, the volume in mm^3 will be returned.
            verbose: whether to execute print statements
        """

        bb = {}

        # perform morphological operations
        if not (self.mm_to_erode and self.mm_to_dilate):
            bb = self.segs

        elif self.mm_to_erode:
            print("Eroding segmentations...") if verbose else None
            for segmentation in self.segs_subset:
                bb[segmentation] = morphology.erode_segmentation(
                    self.segs[segmentation],
                    mm_to_erode=self.mm_to_erode,
                    use_gpu=True
                )
        elif self.mm_to_dilate:
            print("Dilating segmentations...") if verbose else None
            for segmentation in self.segs_subset:
                bb[segmentation] = morphology.dilate_segmentation(
                    self.segs[segmentation],
                    mm_to_dilate=self.mm_to_dilate,
                    use_gpu=True
                )

        # merge (get the set union) the segmentations (aka bounding boxes)
        print("Merging bounding boxes...") if verbose else None
        if isinstance(self.segs_subset, list):
            # remove first list item from bb dictionary
            BB: nib.nifti1.Nifti1Image = bb.pop(self.segs_subset[0])
        elif isinstance(self.segs_subset, str):
            BB: nib.nifti1.Nifti1Image = bb.pop(self.segs_subset)
        for bb_ in bb.values():
            BB: nib.nifti1.Nifti1Image = self._get_set_plus(
                BB, bb_, compute_number=False)
        self.BB = BB

        self.voxels = self.BB.get_fdata()
        if compute_number:
            return np.count_nonzero(self.voxels) * self.voxel_vol
        return self.BB

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
        Plot heatmaps of the sum along each axis of the merged segmentations
        """
        if self.BB is None:
            raise ValueError(
                "Septum segmentation not computed yet - run .compute_septum_volume()")
        merged_segs: np.ndarray = self.BB.get_fdata()

        _, axes = plt.subplots(1, 3, figsize=(18, 6))

        sns.heatmap(np.sum(merged_segs, axis=0), ax=axes[0], cmap='viridis')
        axes[0].set_title('Heatmap: Sum along x-axis')

        sns.heatmap(np.sum(merged_segs, axis=1), ax=axes[1], cmap='viridis')
        axes[1].set_title('Heatmap: Sum along y-axis')

        sns.heatmap(np.sum(merged_segs, axis=2), ax=axes[2], cmap='viridis')
        axes[2].set_title('Heatmap: Sum along z-axis')

        plt.show()

    def vizz_volume(self) -> None:
        """
        Visualize the uptake volume
        """
        if self.BB is None:
            raise ValueError(
                "Septum segmentation not computed yet - run .compute_septum_volume()")
        utils.seg_vizz(
            imgs=[self.BB],
            names=[f'Uptake Volume of {" - ".join(self.segs_subset)}'])
