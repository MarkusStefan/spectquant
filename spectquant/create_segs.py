"""
Automatically creating segmentations with TotalSegmentator
"""

from pathlib import Path
from typing import List, Union, Literal
from totalsegmentator.python_api import totalsegmentator


def create_segs(input_path: str,
                output_path: str,
                task: str = Literal["heartchambers_highres",
                                    "tissue_types", "total"],
                body_seg: bool = True,
                roi_subset: Union[str, List[str]] = None,
                output_type: str = "nifti") -> Union[None, bool]:
    """Create segmentations with TotalSegmentator

    Args:
        input_path (str): _description_
        output_path (str): _description_
        task (str, optional): _description_.
            Defaults to Literal["heartchambers_highres",
                "tissue_types", "total"].
        body_seg (bool, optional): _description_. Defaults to True.
        roi_subset (List[str], optional): _description_.
            Defaults to Literal['heart', 'spinal'].
        output_type (str, optional): _description_. Defaults to "nifti".

    Returns:
        Union[None, bool]: _description_
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    if task == 'total' and roi_subset is None:
        roi_subset = [
            'heart',
            'inferior_vena_cava',
            'autochthon_right',
            'vertebrae_T7',
            'vertebrae_T8',
            'vertebrae_T9',
            'vertebrae_T10',
            'vertebrae_T11']
    elif task != "total":
        roi_subset = None
    elif isinstance(roi_subset, list):
        if len(list) == 0:
            raise ValueError("No empty list allowed for roi_subset")
        roi_subset = roi_subset
    elif isinstance(roi_subset, str):
        roi_subset = [roi_subset]
    else:
        raise ValueError(
            "roi_subset must be a list of strings \
                or a string with value 'heart' or 'spinal'")

    try:
        if roi_subset is not None:
            totalsegmentator(input=input_path,
                             output=output_path,
                             task=task,
                             body_seg=body_seg,
                             roi_subset=roi_subset,
                             output_type=output_type
                             )
        else:
            totalsegmentator(input=input_path,
                             output=output_path,
                             task=task,
                             body_seg=body_seg,
                             output_type=output_type
                             )
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error encounterd with {input_path}")
        return False
