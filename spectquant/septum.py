"""
Class for automatic computation of heart septum volume
"""
try:
    from typing import Optional, Dict, List, Union
except ImportError:
    from collections.abc import Optional, Dict, List, Union

import os
import gc
import numpy as np
import nibabel as nib

from spectquant import utils
from spectquant import morphology

class SeptumVol:

    def __init__(self,
                 segs: Optional[Dict[str, nib.nifti1.Nifti1Image]] = None,
                 segs_path: str = None,
                 seg_subset_names: List[str] = [
                     'heart_ventricle_left', 'heart_ventricle_right'
                 ],
                 mm_to_dilate: Union[int, float] = 20,
                 verbose: bool = False) -> None:

        self.segs = segs
        self.segs_path = segs_path
        self.seg_subset_names = seg_subset_names  # subset of segmentations to load
        self.mm_to_dilate = mm_to_dilate
        self.verbose = verbose
        self.vol3d_available = False

        self.affine = None
        self.LV = None
        self.RV = None
        self.RV_data = None
        self.LV_data = None
        self.RV_dilated = None
        self.LV_dilated = None
        self.RV_dilated_data = None
        self.LV_dilated_data = None
        self.indices_LV = None
        self.indices_RV = None
        self.x_indices_LV = None
        self.x_indices_RV = None
        self.highest_x_index_RV = None
        self.lowest_x_index_LV = None
        self.z_indices_LV = None
        self.lowest_z_index_LV = None
        self.highest_z_index_LV = None
        self.RV_processed = None
        self.LV_processed = None
        self.septum = None

        if self.segs is None and self.segs_path is None:
            raise ValueError(
                "Either segmentations or path to segmentations must be provided")

        if self.segs is None and self.segs_path is not None:
            self.segs = {
                f.split('.')[0]: nib.load(os.path.join(self.segs_path, f))
                for f in os.listdir(self.segs_path)
                if f.endswith('.nii.gz') and f.split('.')[0] in self.seg_subset_names
            }

        if self.segs is not None:
            self.affine = self.segs[self.seg_subset_names[0]].affine

    def _free_memory(self, verbose: bool = False) -> None:
        """
        Free memory by deleting ALL class variables.
        """
        for class_var in self.__dict__.keys():
            print(f"Deleting {class_var}...") if verbose else None
            del class_var
        gc.collect()

    def _load_data(self) -> None:
        """
        Load the data from the segmentations
        """
        self.LV = self.segs['heart_ventricle_left']
        self.RV = self.segs['heart_ventricle_right']
        self.LV_data = self.LV.get_fdata()
        self.RV_data = self.RV.get_fdata()

    def _get_bounding_box_voxel_locs(self) -> None:
        """
        Get the bounding voxel index locations from the *unmmodified* segmentations
        """
        # x_indices_LV = np.any(LV_data, axis=(1, 2))
        # x_indices_RV = np.any(RV_data, axis=(1, 2))
        # highest_x_index_RV = np.where(x_indices_RV)[0].max()
        # lowest_x_index_LV = np.where(x_indices_LV)[0].min()
        # z_indices_LV = np.any(LV_data, axis=(0, 1))
        # lowest_z_index_LV = np.where(z_indices_LV)[0].min()
        # highest_z_index_LV = np.where(z_indices_LV)[0].max()
        self.x_indices_LV = np.any(self.LV_data, axis=(1, 2))
        self.x_indices_RV = np.any(self.RV_data, axis=(1, 2))
        self.highest_x_index_RV = np.where(self.x_indices_RV)[0].max()
        self.lowest_x_index_LV = np.where(self.x_indices_LV)[0].min()
        self.z_indices_LV = np.any(self.LV_data, axis=(0, 1))
        self.lowest_z_index_LV = np.where(self.z_indices_LV)[0].min()
        self.highest_z_index_LV = np.where(self.z_indices_LV)[0].max()

    def _dilate_segmentations(self, verbose: bool = False) -> None:
        """
        Dilate the segmentations
        """
        # copy_LV = nib.nifti1.Nifti1Image(self.LV.get_fdata(), self.LV.affine)
        # copy_RV = nib.nifti1.Nifti1Image(self.RV.get_fdata(), self.RV.affine)
        print("Dilating segmentations") if (verbose or self.verbose) else None
        print(f"np.sum(self.LV_data): {np.sum(self.LV_data)}") if (
            verbose or self.verbose) else None
        print(f"np.sum(self.RV_data): {np.sum(self.RV_data)}") if (
            verbose or self.verbose) else None
        self.LV_dilated = morphology.dilate_segmentation(
            self.LV, self.mm_to_dilate, use_gpu=True)
        self.RV_dilated = morphology.dilate_segmentation(
            self.RV, self.mm_to_dilate, use_gpu=True)
        self.LV_dilated_data = self.LV_dilated.get_fdata()
        self.RV_dilated_data = self.RV_dilated.get_fdata()
        print(f"np.sum(self.LV_dilated_data): {np.sum(self.LV_dilated_data)}") if (
            verbose or self.verbose) else None
        print(f"np.sum(self.RV_dilated_data): {np.sum(self.RV_dilated_data)}") if (
            verbose or self.verbose) else None

    def _cut_dilated_segs(self, verbose: bool = False) -> None:
        """
        Cut the dilated segmentations to the bounding box of the original *unmodified* segmentations
        """
        if verbose:
            print(f"highest_x_index_RV: {self.highest_x_index_RV}")
            print(f"lowest_x_index_LV: {self.lowest_x_index_LV}")
            print(f"lowest_z_index_LV: {self.lowest_z_index_LV}")
            print(f"highest_z_index_LV: {self.highest_z_index_LV}")
            print(f"RV_dilated_data shape: {self.RV_dilated_data.shape}")
            print(f"LV_dilated_data shape: {self.LV_dilated_data.shape}")

        RV_masked = np.zeros_like(self.RV_dilated_data)
        print(
            f"RV_masked.shape: {RV_masked.shape}\nnp.sum(RV_masked): {np.sum(RV_masked)}\n np.sum(self.RV_dilated_data): {np.sum(self.RV_dilated_data)}"
        ) if (verbose or self.verbose) else None

        RV_masked[self.lowest_x_index_LV:, :, self.lowest_z_index_LV:self.highest_z_index_LV +
                  1] = self.RV_dilated_data[self.lowest_x_index_LV:, :, self.lowest_z_index_LV:self.highest_z_index_LV + 1]
        print(f"RV_masked.shape: {RV_masked.shape}\nnp.sum(RV_masked): {np.sum(RV_masked)}") if (
            verbose or self.verbose) else None

        LV_masked = np.zeros_like(self.LV_dilated_data)
        LV_masked[:self.highest_x_index_RV + 1, :,
                  :] = self.LV_dilated_data[:self.highest_x_index_RV + 1, :, :]
        print(f"LV_masked.shape: {LV_masked.shape}\nnp.sum(LV_masked): {np.sum(LV_masked)}") if (
            verbose or self.verbose) else None

        self.RV_processed = nib.nifti1.Nifti1Image(RV_masked, self.RV.affine)
        self.LV_processed = nib.nifti1.Nifti1Image(LV_masked, self.LV.affine)
        print(f"np.sum(self.RV_processed.get_fdata()): {np.sum(self.RV_processed.get_fdata())}") if (
            verbose or self.verbose) else None
        print(f"np.sum(self.LV_processed.get_fdata()): {np.sum(self.LV_processed.get_fdata())}") if (
            verbose or self.verbose) else None

    @staticmethod
    def _get_intersection(setA: np.ndarray | nib.nifti1.Nifti1Image,
                          setB: np.ndarray | nib.nifti1.Nifti1Image,
                          compute_number: bool = True) -> np.ndarray | nib.nifti1.Nifti1Image:
        """
        Get intersecting voxels of two segmentations
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

        # boolean values
        intersection = np.logical_and(setA_, setB_)

        if compute_number:
            return np.count_nonzero(intersection)

        if isinstance(affine, np.ndarray):
            # needs conversion to int
            # print(intersection)
            return nib.nifti1.Nifti1Image(intersection.astype(int), affine)
        return intersection

    @staticmethod
    def _get_set_minus(setA: np.ndarray | nib.nifti1.Nifti1Image,
                       setB: np.ndarray | nib.nifti1.Nifti1Image,
                       compute_number: bool = True) -> np.ndarray | nib.nifti1.Nifti1Image:
        """
        Get the set difference of voxels of two segmentations
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

        set_minus = np.logical_and(setA_, np.logical_not(setB_).astype(int))

        if compute_number:
            return np.count_nonzero(set_minus)

        if isinstance(affine, np.ndarray):
            return nib.nifti1.Nifti1Image(set_minus.astype(int), affine)
        return set_minus

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

    def compute_septum_volume(self,
                              compute_number: bool = True,
                              verbose: bool = False) -> float | nib.nifti1.Nifti1Image:
        """
        Compute septum volume

        Args:
            compute_number: Computes volume as number by default.
                If False, then returns a Nifti1Image object.
            verbose: Whether to print progress.

        """
        self._load_data()
        print("Data loaded") if (verbose or self.verbose) else None
        self._get_bounding_box_voxel_locs()
        print("Bounding box voxel locations computed") if (
            verbose or self.verbose) else None
        self._dilate_segmentations()
        print("Segmentations dilated") if (verbose or self.verbose) else None
        self._cut_dilated_segs()
        print("Dilated segmentations cut") if (
            verbose or self.verbose) else None

        # if compute_number is True, variables are number,
        # else, they are Nifti1Image objects
        print("Computing septum volume as number") if verbose and compute_number else None
        intersection_RVd_LVd = self._get_intersection(
            self.RV_processed, self.LV_processed, compute_number=compute_number)
        print("intersection_RVd_LVd = {}".format(intersection_RVd_LVd)
              ) if verbose and compute_number else None
        intersection_RV_LVd = self._get_intersection(
            self.RV, self.LV_processed, compute_number=compute_number)
        print("intersection_RV_LVd = {}".format(intersection_RV_LVd)
              ) if verbose and compute_number else None
        intersection_LV_RVd = self._get_intersection(
            self.LV, self.RV_processed, compute_number=compute_number)
        print("intersection_LV_RVd = {}".format(intersection_LV_RVd)
              ) if verbose and compute_number else None

        if compute_number:
            self.vol3d_available = False
            excess_dilation = intersection_RV_LVd + intersection_LV_RVd
            nr_septum_voxels = intersection_RVd_LVd - excess_dilation
            voxel_volume = np.prod(self.LV.header.get_zooms())  # in mm^3
            print(
                "voxel_volume = {}".format(voxel_volume)) if (
                verbose or self.verbose) else None
            septum_volume = nr_septum_voxels * voxel_volume
            return septum_volume

        print("Septum volume as Nifti1Image object") if (
            verbose or self.verbose) else None
        excess_dilation = self._get_set_plus(
            intersection_RV_LVd,
            intersection_LV_RVd,
            compute_number=compute_number)
        self.septum = self._get_set_minus(
            intersection_RVd_LVd,
            excess_dilation,
            compute_number=compute_number)
        self.vol3d_available = True
        return self.septum

    def vizz_septum(self) -> None:
        """
        Visualize the septum
        """
        if self.septum is None:
            raise ValueError(
                "Septum segmentation 3d volume not computed yet - run .compute_septum_volume(compute_number=False)")
        utils.seg_vizz(
            imgs=[self.RV, self.LV, self.septum],
            names=['RV', 'LV', 'septum']
        )
