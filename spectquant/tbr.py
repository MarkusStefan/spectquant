"""
Isolated instance for automating TBR computation - very similar to SUV computation
"""
try:
    from typing import Optional, List, Dict, Union
except (ImportError, ModuleNotFoundError):
    from collections.abc import Optional, List, Dict, Union

import nibabel as nib

from spectquant import morphology
from spectquant.suv import SUV


class TBR(SUV):

    """
    A class for computing Targent Background/Blood Ratio (TBR) by inheriting from SUV class.

    The SUV class is designed to handle the computation of SUV metrics from medical imaging data,
    specifically CT and SPECT scans. It provides methods to process and analyze segmented regions
    of interest (ROIs) within the scans.

    Attributes:
        segs (Optional[Dict[str, nib.nifti1.Nifti1Image]]): Dictionary of CT scan segmentations.
        spect (nib.nifti1.Nifti1Image): The SPECT scan image.
        ct (Optional[nib.nifti1.Nifti1Image]): The CT scan image.
        segs_path (str): Path to the segmentation files - used if ct is not provided.
        spect_path (str): Path to the SPECT scan file - used if spect is not provided.
        ct_path (str): Path to the CT scan file - used for creating segmentations.
        segs_subset (Union[str, List[str]]): Subset of segmentations to use.
        threshold (Union[int, float]): Threshold value for segmentation.
        mm_to_erode (Union[int, float]): Millimeters to erode the segmentation.
            Default is to erode the segmentation.
        mm_to_dilate (Union[int, float]): Millimeters to dilate the segmentation.
            If provided, will dilate the segmentation instead of eroding.
        suv_cube_vol (Union[int, float]): Volume of the cube for SUV calculation.
        suv_method (str): Method for SUV calculation ('peak', 'mean', etc.).
        use_convolution (bool): Whether to use convolution for SUV calculation.
        use_gpu (bool): Whether to use GPU for computations.
        verbose (bool): Whether to print detailed logs.

    Methods:

        _create_segs(input_path: str = '', output_path: str = '') -> None:
            Create segmentations from CT scans.

        _free_memory(verbose: bool = False) -> None:
            Free memory by deleting all class variables.

        _preprocess(img: nib.nifti1.Nifti1Image) -> nib.nifti1.Nifti1Image:
            Preprocess the image.

        compute_suv(roi: str) -> float:
            Computes the SUV for a given region of interest (ROI).

        compute_spect_suv(roi: str, ref_roi: str) -> float:
            Computes the SUV from the SPECT scan for a given ROI using a reference ROI.

        compute_retention_index(roi: str, time_points: List[float]) -> float:
            Computes the retention index for a given ROI over specified time points.
    """

    def __init__(self,
                 segs: Optional[Dict[str, nib.nifti1.Nifti1Image]] = None,
                 spect: nib.nifti1.Nifti1Image = None,
                 ct: Optional[nib.nifti1.Nifti1Image] = None,
                 segs_path: str = '',
                 spect_path: str = '',
                 ct_path: str = '',
                 segs_subset: Union[str, List[str]] = None,
                 threshold: Union[int, float] = 0.4,
                 mm_to_erode: Union[int, float] = 3,
                 mm_to_dilate: Union[int, float] = None,
                 tbr_cube_vol: Union[int, float] = 1,
                 tbr_method: str = 'peak',
                 use_convolution: bool = True,
                 use_gpu: bool = False,
                 verbose: bool = True) -> None:

        # pass every argument to parent class -> TBR to SUV
        super().__init__(segs=segs,
                         spect=spect,
                         ct=ct,
                         segs_path=segs_path,
                         spect_path=spect_path,
                         ct_path=ct_path,
                         segs_subset=segs_subset,
                         threshold=threshold,
                         mm_to_erode=mm_to_erode,
                         mm_to_dilate=mm_to_dilate,
                         suv_cube_vol=tbr_cube_vol,
                         suv_method=tbr_method,
                         use_convolution=use_convolution,
                         use_gpu=use_gpu,
                         verbose=verbose)

        # load Vena Cava for TBR computation -> adjusting SUV by mean uptake of
        # blood
        self.vena_cava = self._preprocess(self.segs['inferior_vena_cava'])
        # derive mean SUV from vena cava using SUV mean computation (only from
        # non-zero voxels)
        self.vena_cava_mean, _ = morphology.compute_suv(self.spect,
                                                        self.vena_cava,
                                                        cube_vol=self.suv_cube_vol,
                                                        method='mean',
                                                        use_convolution=self.use_convolution,
                                                        use_gpu=self.use_gpu)

    def _create_segs(self, input_path: str = '',
                     output_path: str = '') -> None:
        return super()._create_segs(input_path=input_path,
                                    output_path=output_path)

    def _free_memory(self, verbose: bool = False) -> None:
        return super()._free_memory(verbose=verbose)

    def _preprocess(self, img: nib.Nifti1Image) -> nib.nifti1.Nifti1Image:
        return super()._preprocess(img=img)

    # def compute_tbr(self, *args, **kwargs) -> float:
    #     # Call the parent class compute_suv method
    #     suv_value = super().compute_suv(*args, **kwargs)
    #     tbr_value = float( suv_value / self.vena_cava_mean )
    #     return tbr_value

    def compute_tbr(self,
                    body_part: str = 'heart_myocardium',
                    preprocess: bool = True) -> float:
        """
        Compute TBR (Target to Blood Ratio) from CT segmentation masks.
        Args:
            body_part: Which body part to use SPECT and segmentation for computing TBR - \
                use naming from TotalSegmentator.
            preprocess: Whether to preprocess or not.
        Returns:
            Computet TBR value as float number
        """
        # enforce erosion for CT segmentations
        # -> adjust for possible false positives from segmentation mask
        if not self.erode:
            raise ValueError(".compute_tbr() requires self.erode=True \
                             - dilation is not supported for SPECT segmentations!")
        if self.mm_to_erode is None:
            raise ValueError(".compute_tbr() requires mm_to_erode to be set \
                             - dilation is not supported for SPECT segmentations!")

        # Utilize compute_suv method of parent class
        suv_value = super().compute_suv(body_part=body_part,
                                        preprocess=preprocess)
        # adjust for average blood uptake
        tbr_value = float(suv_value / self.vena_cava_mean)
        return tbr_value

    def compute_spect_tbr(self,
                          body_part: str = 'heart_myocardium',
                          ref_roi: str = 'heart',
                          mode_ref_roi: str = 'peak',
                          preprocess: bool = True) -> float:
        """
        Compute TBR from SPECT segmentations by using thresholding and dilated segmentations
            as bounding-box.
        Args:
            body_part: Which body part to use SPECT and segmentation for computing TBR.
                Use naming from TotalSegmentator.
            peak_roi: Which body part to use for TBR peak value computation.
                Typically, this is the entire heart.
            mode_ref_roi: Which method to use for computing TBR from reference ROI.
            preprocess: Whether to preprocess or not.
        Returns:
            Computed TBR value as float number
        """
        if self.erode:
            raise ValueError(".compute_spect_tbr() requires self.erode=False \
                             - erosion is not supported for SPECT segmentations!")
        if not self.mm_to_dilate:
            # enforce dilation for SPECT segmentations
            # -> ROI are spread out compared to actual CT delineations
            raise ValueError(".compute_spect_tbr() requires mm_to_dilate to be set \
                             - erosion is not supported for SPECT segmentations!")

        # Utilize compute_spect_suv method of parent class
        # NOTE: uses SUV_{mode_ref_roi} for roi determination
        suv_value = super().compute_spect_suv(body_part=body_part,
                                              ref_roi=ref_roi,
                                              mode_ref_roi=mode_ref_roi,
                                              preprocess=preprocess)
        # adjust for average blood uptake
        tbr_value = float(suv_value / self.vena_cava_mean)
        return tbr_value


    # def compute_retention_idx(self,
    #                           vertebrae: str | int = "T9",
    #                           use_median_vertebrae: bool = False) -> float:
    #     suv_rtidx = super().compute_retention_idx(vertebrae=vertebrae,
    #                                   use_median_vertebrae=use_median_vertebrae)
    #     tbr_rtidx = float(suv_rtidx / self.vena_cava_mean) # adjust for average blood uptake
    #     return
