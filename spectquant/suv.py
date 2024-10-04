"""
Isolated instance for automating SUV computation
"""
import os
import gc
from typing import Optional, List, Dict, Union

import numpy as np
import nibabel as nib
    
from spectquant import utils
from spectquant import morphology
from spectquant.create_segs import create_segs


class SUV:
    
    """
    A class for computing Standardized Uptake Values (SUV) and retention index.

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
                 suv_cube_vol: Union[int, float] = 1,
                 suv_method: str = 'peak',
                 use_convolution: bool = True,
                 use_gpu: bool = False,
                 verbose: bool = True) -> None:

        self.ct = ct
        self.segs = segs
        self.spect = spect
        self.ct_path = ct_path
        self.segs_path = segs_path
        self.spect_path = spect_path
        self.segs_subset = segs_subset
        self.threshold = threshold
        self.mm_to_erode = mm_to_erode
        self.mm_to_dilate = mm_to_dilate
        self.suv_cube_vol = suv_cube_vol
        self.suv_method = suv_method
        self.use_convolution = use_convolution
        self.use_gpu = use_gpu
        self.verbose = verbose

        if self.segs_subset is None:
            self.segs_subset = [''] # assign empty list

        if (self.segs is None and self.ct is None) and \
                (self.ct_path == '' and self.segs_path == ''):
            raise ValueError("Either CT or segmentations must be provided")
        if self.spect is None and self.spect_path == '':
            raise ValueError("SPECT scan must be provided")
        if self.ct_path != '':
            self.ct = nib.load(self.ct_path)
        if self.segs_path != '':
            self.segs = {
                f.split('.')[0]: nib.load(os.path.join(self.segs_path, f))
                for f in os.listdir(self.segs_path) if f.endswith('.nii.gz')
            }
        if self.spect_path != '':
            self.spect = nib.load(self.spect_path)

        if mm_to_dilate:  # whether to use erosion or dilation
            self.erode = False
        else:
            self.erode = True

    def _create_segs(self, input_path: str = '', output_path: str = '') -> None:
        """
        Create segmentations from CT scans.
        """
        if self.segs is not None:
            print("Segmentations already provided")
            return None

        if self.ct is None:
            raise ValueError("CT scan must be provided")

        if input_path == '' and self.ct_path != '':
            input_path = self.ct_path
        else:
            raise ValueError("Input path must be provided")
        if output_path == '' and self.segs_path != '':
            output_path = self.segs_path
        else:
            raise ValueError("Output path must be provided")

        spinal = ['inferior_vena_cava', 'autochthon_right', 'vertebrae_T7', 'vertebrae_T8',
                  'vertebrae_T9', 'vertebrae_T10', 'vertebrae_T11']
        heart = ['heart', 'heart_myocardium', 'heart_atrium_left', 'heart_ventricle_left',
                 'heart_atrium_right', 'heart_ventricle_right', 'aorta', 'pulmonary_artery']

        if isinstance(self.segs_subset, list) and len(self.segs_subset) == 1:
            self.segs_subset = spinal + \
                ['heart_myocardium']  # only myocardium needed
        elif isinstance(self.segs_subset, list) and len(self.segs_subset) > 2:
            self.segs_subset = spinal + heart  # just load all
        elif isinstance(self.segs_subset, str):
            if self.segs_subset == 'spinal':
                self.segs_subset = spinal
            elif self.segs_subset == 'heart':
                self.segs_subset = heart
            else:
                self.segs_subset = [self.segs_subset]
        else:
            raise ValueError("Invalid segs_subset")

        # create spinal segmentations
        create_segs(input_path=input_path,
                    output_path=output_path,
                    task="total",
                    roi_subset=spinal,
                    body_seg=True,
                    output_type="nifti")
        # create heart segmentations
        create_segs(input_path=input_path,
                    output_path=output_path,
                    task="heartchambers_highres",
                    body_seg=True,
                    output_type="nifti")
        self.segs = {
            f.split('.')[0]: nib.load(os.path.join(output_path, f))
            for f in os.listdir(output_path)
            if f.split('.')[0] in self.segs_subset and f.endswith('.nii.gz')
        }

        return None


# 1 load t7 to t11, autochthon_right and myocardium and erode/dilate the segmentations
# - erosion -> SUV within CT segmentation mask
# - dilation -> SUV from SPECT activation / outside segmentation mask
# 2 cut autochthon right by t7 to t11
# 3 compute suv peak for each {cut-autochthon right, t{xx}, myocardium}
# 4 compute retention index

    def _free_memory(self, verbose: bool = False) -> None:
        """
        Free memory by deleting ALL class variables.
        """
        for class_var in self.__dict__:
            print(f"Deleting {class_var}...") if verbose else None
            del class_var
        gc.collect()

    def _preprocess(self, img: nib.nifti1.Nifti1Image) -> nib.nifti1.Nifti1Image:
        """
        Preprocess image.
        """
        if self.erode:
            eroded_or_dilated = morphology.erode_segmentation(img,
                                                              self.mm_to_erode,
                                                              use_gpu=self.use_gpu)
        else:
            eroded_or_dilated = morphology.dilate_segmentation(img,
                                                               self.mm_to_dilate,
                                                               use_gpu=self.use_gpu)
        cleaned = utils.keep_above_mean_components(eroded_or_dilated)
        return cleaned

    def compute_suv(self,
                    body_part: str = 'heart_myocardium',
                    preprocess: bool = True) -> float:
        """
        Compute SUV.
        Args:
            body_part: Which body part to use SPECT and segmentation for computing SUV
                Use naming from TotalSegmentator.
            preprocess: Whether to preprocess or not.
        Returns:
            Computet Standard Uptake Value as float number
        """
        if not self.erode:
            # enforce erosion for CT segmentations -> ROI are more precise than
            # SPECT segmentations
            raise ValueError(".compute_suv() requires mm_to_dilate to be set to None (default) - \
                             dilation is not supported for CT segmentations!")
        if self.mm_to_erode is None:
            raise ValueError(".compute_suv() requires mm_to_erode to be set - \
                             dilation is not supported for CT segmentations!")
        seg = self._preprocess(
            self.segs[body_part]) if preprocess else self.segs[body_part]
        suv, _ = morphology.compute_suv(self.spect,
                                        seg,
                                        cube_vol=self.suv_cube_vol,
                                        method=self.suv_method,
                                        use_convolution=self.use_convolution,
                                        use_gpu=self.use_gpu)
        return float(suv)

    def compute_spect_suv(self,
                          body_part: str = 'heart_myocardium',
                          ref_roi: str = 'heart',
                          mode_ref_roi: str = 'peak',
                          preprocess: bool = True) -> float:
        """
        Compute SUV from SPECT segmentations by using thresholding and dilated segmentations
            as bounding-box.
        Args:
            body_part: Which body part to use SPECT and segmentation for computing TBR.
                Use naming from TotalSegmentator.
            peak_roi: Which body part to use for SUV peak value computation.
                Typically, this is the entire heart.
            mode_ref_roi: Which method to use for computing SUV from reference ROI.
            preprocess: Whether to preprocess or not.
        Returns:
            Computet SUV value as float number
        """
        if self.erode:
             raise ValueError(".compute_spect_suv() requires self.erode=False \
                             - dilation is not supported for SPECT segmentations!")
        if self.mm_to_dilate is None:
            raise ValueError(".compute_spect_suv() requires mm_to_dilate to be set \
                             - dilation is not supported for SPECT segmentations!")

        seg = self._preprocess(
            self.segs[body_part]) if preprocess else self.segs[body_part]

        ref_roi_processed = self._preprocess(self.segs[ref_roi]) if preprocess \
            else self.segs[ref_roi]
        ref_roi_suv, _ = morphology.compute_suv(self.spect,
                                                ref_roi_processed,
                                                cube_vol=self.suv_cube_vol,
                                                method=mode_ref_roi,
                                                use_convolution=self.use_convolution,
                                                use_gpu=self.use_gpu)
        if not isinstance(ref_roi_suv, (int, float, np.number)):
            ref_roi_suv = ref_roi_suv.get()  # infer that ref_roi_suv is a cupy.ndarray

        mask = self.spect.get_fdata() > (ref_roi_suv * self.threshold)
        thresholded_spect: np.ndarray = self.spect.get_fdata() * mask
        thresholded_spect_nifti = nib.Nifti1Image(thresholded_spect, self.spect.affine)

        suv, _ = morphology.compute_suv(thresholded_spect_nifti,
                                        seg,
                                        cube_vol=self.suv_cube_vol,
                                        method=self.suv_method,
                                        use_convolution=self.use_convolution,
                                        use_gpu=self.use_gpu)
        return float(suv)

    def compute_retention_idx(self,
                              vertebrae: Union[str, int] = "T9",
                              use_median_vertebrae: bool = False) -> float:
        """
        Compute retention index.
        """
        if isinstance(vertebrae, int):
            vertebrae = f"T{vertebrae}"

        myocardium = self.segs['heart_myocardium']
        t7 = self._preprocess(self.segs['vertebrae_T7'])

        # additional error handling for T11
        # -> in case of incomplete segmentation, T11 might not be available
        try:
            t11 = self._preprocess(self.segs['vertebrae_T11'])
        except ValueError:
            # use t10 instead of t11 to cut autochthon muscle
            t11 = self._preprocess(self.segs['vertebrae_T10'])

        if use_median_vertebrae:
            # select vertebrae with median SUV_peak to reduce change of capturing inlfated uptakes
            # due to degenerative changes
            t8 = self._preprocess(self.segs['vertebrae_T8'])
            t9 = self._preprocess(self.segs['vertebrae_T9'])
            t10 = self._preprocess(self.segs['vertebrae_T10'])
            suv_t8, _ = morphology.compute_suv(self.spect,
                                               t8,
                                               cube_vol=self.suv_cube_vol,
                                               method=self.suv_method,
                                               use_convolution=self.use_convolution,
                                               use_gpu=self.use_gpu)
            suv_t9, _ = morphology.compute_suv(self.spect,
                                               t9,
                                               cube_vol=self.suv_cube_vol,
                                               method=self.suv_method,
                                               use_convolution=self.use_convolution,
                                               use_gpu=self.use_gpu)
            suv_t10, _ = morphology.compute_suv(self.spect,
                                                t10,
                                                cube_vol=self.suv_cube_vol,
                                                method=self.suv_method,
                                                use_convolution=self.use_convolution,
                                                use_gpu=self.use_gpu)
            suv_peaks = {'T8': suv_t8, 'T9': suv_t9, 'T10': suv_t10}
            vals = list(suv_peaks.values())
            keys = list(suv_peaks.keys())
            if self.use_gpu:
                vals = [v.get() for v in vals]  # convert from cupy to np
            median_suv_peak = np.median(vals)
            idx = np.argmin(np.abs(vals - median_suv_peak))
            tx = keys[idx]
            if tx == 'T8':
                suv_vertebral = suv_t8
            elif tx == 'T9':
                suv_vertebral = suv_t9
            elif tx == 'T10':
                suv_vertebral = suv_t10
            else:
                raise ValueError("Invalid vertebrae -> \
                                 no vertebrae selected for retention index computation")

        else:
            # use specified vertebrae
            t_vertebrae = self._preprocess(self.segs[f'vertebrae_{vertebrae}'])
            suv_vertebral, _ = morphology.compute_suv(self.spect,
                                                      t_vertebrae,
                                                      ube_vol=self.suv_cube_vol,
                                                      method=self.suv_method,
                                                      use_convolution=self.use_convolution,
                                                      use_gpu=self.use_gpu)

        autochthon_right = self.segs['autochthon_right']
        # only keep segmentaition mask between T7 and T11 to limit search space
        # NOTE: uses T10 if T11 is incomplete and woould otherwise cause a
        # ZeroDivisionError
        autochthon_right_trimmed = utils.trim_seg_z(
            autochthon_right, upper_img=t7, lower_img=t11)

        to_preprocess = [myocardium, autochthon_right_trimmed]
        preprocessed = [self._preprocess(img) for img in to_preprocess]

        suv_cardiac, _ = morphology.compute_suv(self.spect,
                                                preprocessed[0],
                                                cube_vol=self.suv_cube_vol,
                                                method=self.suv_method,
                                                use_convolution=self.use_convolution,
                                                use_gpu=self.use_gpu)
        suv_paraspinal_muscle, _ = morphology.compute_suv(self.spect,
                                                          preprocessed[1],
                                                          cube_vol=self.suv_cube_vol,
                                                          method=self.suv_method,
                                                          use_convolution=self.use_convolution,
                                                          use_gpu=self.use_gpu)

        if self.verbose:
            print('Cardiac', suv_cardiac)
            print('Vertebral', suv_vertebral)
            print('Paraspinal muscle', suv_paraspinal_muscle)

        return float((suv_cardiac / suv_vertebral) * suv_paraspinal_muscle)
