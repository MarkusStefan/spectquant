""" 
Module for utility functions frequently used in the NUKAI project.
"""
import os
try:
    from typing import Union, List, Tuple, Optional
except (ImportError, ModuleNotFoundError):
    from collections.abc import Union, List, Tuple, Optional
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from scipy import ndimage
from nilearn import image
import nibabel as nib

# ignore continuous interpolation warning in nilearn.image.resample_img
warnings.filterwarnings('ignore', category=UserWarning, module='nilearn')


def adjust_voxel_vals_for_suv(spect_csv: Union[str, pd.DataFrame],
                              patient_data_csv: Union[str, pd.DataFrame],
                              output_dir: str,
                              id_col: str = 'ID',
                              spectdir_col: str = 'SPECTDIR') -> nib.nifti1.Nifti1Image:
    """
    Adjusts voxel values in a SPECT scan for SUV calculation.
    Args:
        spect_csv: CSV file containing the patient ID (name+scan_date) + SPECT path.
        patient_data_csv: CSV file containing the patient data.
        patient: Patient ID.
        id_col: Column name of the ID in the CSV file.
        dicomdir_col: Column name of the DICOMDIR in the CSV file.
    """
    def _convert_time_format(time: str):
        # cut off milliseconds
        time = str(time).split(".", maxsplit=1)[0]
        time_format = "%H%M%S"
        time = datetime.strptime(time, time_format)
        return time

    def _assert_parameters(row) -> None:
        # ensure parameter validity
        check = [
            row['weight'],
            row['time_difference'],
            row['half_life'],
            row['injected_dose']]
        for c in check:
            assert c > 0, f"Invalid input. No negative values allowed. Value: {c}"
            assert (
                not np.isnan(c)), f"Invalid input. No NaNs allowed. Value is NaN: {np.isnan(c)}"
        assert row['weight'] < 1000, "Weight exceeds 1000 kg, did you really used kg unit?"

    def _compute_injected_dose_decay(row):
        time_difference = row['time_difference']
        injected_dose = row['injected_dose']
        half_life = row['half_life']
        decay = np.exp(-np.log(2) * time_difference / half_life)
        injected_dose_decay = injected_dose * decay
        return injected_dose_decay

    if isinstance(spect_csv, str):
        spect_df = pd.read_csv(spect_csv)
    elif isinstance(spect_csv, pd.DataFrame):
        spect_df = spect_csv
    else:
        raise ValueError(
            "Input must be a CSV file path or a pandas DataFrame!")

    if isinstance(patient_data_csv, str):
        patient_data_df = pd.read_csv(patient_data_csv)
    elif isinstance(patient_data_csv, pd.DataFrame):
        patient_data_df = patient_data_csv
    else:
        raise ValueError(
            "Input must be a CSV file path or a pandas DataFrame!")

    scan_time = patient_data_df['scan_time'].apply(_convert_time_format)
    injection_time = patient_data_df['injection_time'].apply(
        _convert_time_format)
    time_difference = scan_time - injection_time
    patient_data_df['time_difference'] = time_difference.dt.seconds
    patient_data_df.apply(_assert_parameters, axis=1)  # apply row-wise
    patient_data_df['weight_gramms'] = patient_data_df['weight'] * 1000
    patient_data_df['injected_dose_decay'] = patient_data_df.apply(_compute_injected_dose_decay,
                                                                   axis=1)  # apply row-wise

    # iterrate over every SPECT scan and find matching patient data
    for _, row in spect_df.iterrows():
        spect_filename = row[spectdir_col].rsplit(
            "/")[-1].rstrip('.nii.gz')  # mac & linux
        spect_filename = spect_filename.rsplit(
            '\\')[-1].rstrip('.nii.gz')  # windows
        spect = nib.load(row[spectdir_col])
        img_arr = spect.get_fdata()
        img_arr_T = np.transpose(img_arr)

        # find corresponding patient data
        patient_row = patient_data_df[patient_data_df[id_col] == row[id_col]]
        if patient_row.empty:
            # raise ValueError(f"No patient data found for {row[id_col]}")
            print(f"No patient data found for {row[id_col]}")
        else:
            patient_row = patient_row.iloc[0]
            weight_gramms = patient_row['weight_gramms']
            injected_dose_decay = patient_row['injected_dose_decay']
            suv_map = img_arr_T * weight_gramms / injected_dose_decay
            suv_img = nib.Nifti1Image(
                np.transpose(suv_map), spect.affine, spect.header)
            output_folder = os.path.join(output_dir, f"{row[id_col]}")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)
            nib.save(
                suv_img,
                os.path.join(
                    output_folder,
                    f"{spect_filename}_SUV.nii.gz"))

    return None


def resample_img(img: nib.nifti1.Nifti1Image,
                 resample_to_img: nib.nifti1.Nifti1Image = None,
                 new_shape: Optional[Tuple[int, int, int]] = None,
                 interpolation: str = 'nearest'
                 #  new_voxel_size:Union[List[float],
                 #                       Tuple[float, float, float]]=None
                 ) -> nib.nifti1.Nifti1Image:
    """
    Resamples a nifti image to a new shape and voxel size.

    Args:
        img: nifti image to be resampled.
        resample_to_img: nifti image to resample img to.
        new_shape: new shape of the resampled image.
        new_voxel_size: new voxel size of the resampled image.

    Returns:
        Resampled nifti image.
    """
    warnings.filterwarnings(
        action='always')  # ignore data type casting warnings
    new_affine = None
    if resample_to_img is not None:
        new_affine = resample_to_img.affine

    if new_shape is None:
        new_shape = resample_to_img.header.get_data_shape()

    if resample_to_img is None and new_shape is None:
        raise ValueError(
            "Either 'resample_to_img' or 'new_shape' and 'new_voxel_size' must be supplied.")

    resampled_img = image.resample_img(
        img,
        target_affine=new_affine,
        target_shape=new_shape,
        interpolation=interpolation)
    return resampled_img


def get_seg_components(img: nib.nifti1.Nifti1Image,
                       return_list: bool = False) -> List[np.ndarray]:
    """
    Takes in a nifti image and yields a generator object to iterate over np.ndarrays, 
        each being a connected component.
    Optionally, a list containing each component can be returned.
    """
    labeled_array, num_features = ndimage.label(img.get_fdata())

    def conn_generator(labeled_array, num_features):
        for label in range(1, num_features + 1):
            component = np.where(labeled_array == label, 1, 0)
            yield component

    def conn_list(labeled_array, num_features):
        components = []
        for label in range(1, num_features + 1):
            component = np.where(labeled_array == label, 1, 0)
            components.append(component)
        return components

    if return_list:
        return conn_list(labeled_array, num_features)
    return conn_generator(labeled_array, num_features)


def keep_above_mean_components(
        img: nib.nifti1.Nifti1Image) -> nib.nifti1.Nifti1Image:
    """
    Only keeps components with a volume larger than the arithmetic mean of all components.
    """

    components = get_seg_components(
        img, return_list=True)  # retrieve all components
    volumes = [
        (comp,
         compute_volume(
             nib.Nifti1Image(comp, img.affine, img.header), unit="cm^3"
         )
         ) for comp in components  # compute volumes for components
    ]

    if len(volumes) == 0:
        raise ValueError("No components found in the segmentation image\n\t-> all voxels are 0\n\t\
                         -> check input image!")
    if len(volumes) == 1:
        return img
    average_volume = sum(vol for _, vol in volumes) / \
        len(volumes)  # compute average volume

    # keep only components with volume >= average_volume
    above_average_components = [comp for comp,
                                vol in volumes if vol >= average_volume]

    above_average_mask = np.zeros_like(
        img.get_fdata())  # create 3D object with 0s
    for comp in above_average_components:
        above_average_mask += comp  # add the components to keep to mask

    img_data = img.get_fdata()
    # set the smallest components to 0 in the original image
    img_data[above_average_mask == 0] = 0
    # create a new image with the modified data
    cleaned_img = nib.Nifti1Image(img_data, img.affine, img.header)

    return cleaned_img


def keep_n_largest_components(img: nib.nifti1.Nifti1Image,
                              n_components: Union[int, List[int]]) -> nib.nifti1.Nifti1Image:
    """
    Only keeps the n largest components in a segmentation image.
    """
    n_list: list = n_components  # avoid inplace modification of input list

    for n in n_list:
        components = get_seg_components(
            img, return_list=True)  # retrieve all components
        volumes = [
            (comp,
             compute_volume(
                 nib.Nifti1Image(comp, img.affine, img.header), unit="cm^3")
             ) for comp in components  # compute volumes for components
        ]
        # sort components by volume (largest first)
        volumes.sort(key=lambda x: x[1], reverse=True)

        n_largest_components = volumes[:n]  # keep only first n_components
        # create a mask of n_largest_components
        largest_mask = np.zeros_like(img.get_fdata())
        for comp, _ in n_largest_components:
            largest_mask += comp

        img_data = img.get_fdata()  # extract np.ndarray
        # set the smallest components to 0 in the original image
        img_data[largest_mask == 0] = 0
        cleaned_img = nib.Nifti1Image(
            img_data, img.affine, img.header)  # new img with modified data

        return cleaned_img


def _get_highest_voxel_z(array: np.ndarray) -> int:
    """
    Get the highest voxel index on the z-axis from a segmentation image.
    """
    indices = np.argwhere(array == 1)
    z_coords = indices[:, 2]
    # return max z-coordinate where voxel has value 1
    return np.max(z_coords)


def _get_lowest_voxel_z(array: np.ndarray) -> int:
    """
    Get the lowest voxel index on the z-axis from a segmentation image.
    """
    indices = np.argwhere(array == 1)
    z_coords = indices[:, 2]
    # return min z-coordinate where voxel has value 1
    return np.min(z_coords)


def slice_img(img: Union[nib.nifti1.Nifti1Image, np.ndarray],
              x_: Union[Tuple[int, int], int] = None,
              y_: Union[Tuple[int, int], int] = None,
              z_: Union[Tuple[int, int], int] = None) -> nib.nifti1.Nifti1Image:
    """
    Slices nifti image in all possible dimensions.
    Returns:
        Sliced nifti image with reduced dimensions.
    """
    try:
        mtx = img.get_fdata()
    except AttributeError:
        mtx = img
    if x_ is None:
        x_ = mtx.shape[0]
    if y_ is None:
        y_ = mtx.shape[1]
    if z_ is None:
        z_ = mtx.shape[2]

    if isinstance(x_, int):
        x_ = (0, x_)
    if isinstance(y_, int):
        y_ = (0, y_)
    if isinstance(z_, int):
        z_ = (0, z_)

    # slice array
    mtx = mtx[x_[0]:x_[1], y_[0]:y_[1], z_[0]:z_[1]]
    sliced_img = nib.Nifti1Image(mtx, img.affine, img.header)
    return sliced_img


def trim_seg_z(img: nib.nifti1.Nifti1Image,
               upper_img: nib.nifti1.Nifti1Image,
               lower_img: nib.nifti1.Nifti1Image,
               vizz: bool = False) -> Union[nib.nifti1.Nifti1Image, None]:
    """
    Trimms (without reducing size!) nifti segmentation image on the z-axis according to the highest
        and lowest index position of any other input segmentations.
        Helper functions _get_{highest, lowest}_voxel_z are used.
    Args:
        img: nifti image segmentation to be trimmed.
        upper_img: nifti image segmentation which highest voxel index is used for timming.
        lower_img: nifti image segmentation which lowest voxel index is used for timming.
    Returns:
        vizz=False: trimmed nifti image.
        vizz=True: None.
    """
    upper = _get_highest_voxel_z(upper_img.get_fdata())
    lower = _get_lowest_voxel_z(lower_img.get_fdata())

    data = img.get_fdata().copy()
    # set all voxels above upper and below lower to 0
    data[:, :, :lower] = 0
    data[:, :, upper:] = 0
    trimmed_img = nib.Nifti1Image(data, img.affine, img.header)
    if vizz:
        seg_vizz([img, trimmed_img], size=0.7, opacity=0.5)
        return None

    return trimmed_img


def compute_volume(img: nib.nifti1.Nifti1Image, unit: str = 'mm^3') -> float:
    """
    Computes actual volume of nifti segmentation.
    """
    data = img.get_fdata()
    voxel_sizes = np.sqrt(np.sum(img.affine**2, axis=0))[:3]
    # Compute the volume in mm^3
    volume_mm3 = np.sum(data) * np.prod(voxel_sizes)

    # Convert the volume to the desired unit
    if unit == 'cm^3':
        volume = volume_mm3 / 1000
    elif unit == 'liters':
        volume = volume_mm3 / 1000000
    else:
        volume = volume_mm3

    return volume


def vol_vizz(img: Union[nib.nifti1.Nifti1Image, np.ndarray],
             opacity: Union[int, float] = 0.5) -> None:
    """
    Plots an interactive 3D body. Serves for inspecting shapes.
    """

    if isinstance(img, nib.nifti1.Nifti1Image):
        data = img.get_fdata()
    elif isinstance(img, np.ndarray):
        data = img
    else:
        raise ValueError("Input must be a nifti image or a numpy array!")

    # 3D volume plot
    fig = go.Figure(data=go.Volume(
        x=np.arange(data.shape[0]).repeat(data.shape[1] * data.shape[2]),
        y=np.tile(
            np.arange(
                data.shape[1]).repeat(
                data.shape[2]),
            data.shape[0]),
        z=np.tile(np.arange(data.shape[2]), data.shape[0] * data.shape[1]),
        value=data.flatten(),
        isomin=data.max(),
        isomax=data.max(),
        opacity=opacity,  # opacity levels
        surface_count=1,  # nr of isosurfaces
        colorscale='Viridis'
    ))

    try:
        seg_type = img.get_filename().split('/')[-1].split('.')[0].capitalize()
    except (AttributeError, IndexError):
        seg_type = 'unknown'

    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        title=f'3D Vizz of segmented {seg_type}',
    )

    fig.show()


def static_vol_vizz(img: Union[nib.nifti1.Nifti1Image, np.ndarray]) -> None:
    """
    Plots a static 3D body.
    """

    if isinstance(img, nib.nifti1.Nifti1Image):
        data = img.get_fdata()
    elif isinstance(img, np.ndarray):
        data = img
    else:
        raise ValueError("Input must be a nifti image or a numpy array!")

    voxel_data = data > 0

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the volume
    ax.voxels(voxel_data, edgecolor='k')

    # Set labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()



def seg_vizz(imgs: Union[nib.nifti1.Nifti1Image,
                         np.ndarray,
                         List[Union[nib.nifti1.Nifti1Image, np.ndarray]]
                         ],
             names: List[str] = None,
             size: Union[int, float] = 0.5,
             opacity: Union[int, float] = 0.5) -> None:
    """
    Takes in one or many nifti segmentations and visualizes them together in 3D.
    Segmentation images must be strictly of the same shape.
    """

    if not isinstance(imgs, list):
        imgs = [imgs]

    if names is None:
        names = []
        for c, img in enumerate(imgs):
            if isinstance(img, nib.nifti1.Nifti1Image):
                names.append(img.get_filename().split(
                    '/')[-1].split('.')[0].capitalize())
            elif isinstance(img, np.ndarray):
                names.append('image_' + str(c + 1))
            else:
                raise ValueError(
                    "Input must be a nifti image or a numpy array!")

    elif len(names) != len(imgs):
        raise ValueError(
            "The length of 'names' must match the length of 'imgs'.")

    fig = go.Figure()
    colors = ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
              'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
              'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
              'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
              'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
              'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
              'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
              'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
              'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
              'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
              'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
              'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
              'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
              'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
              'ylorrd']

    # add the reverse of each color scale
    colors = colors + [color + '_r' for color in colors]

    for idx, img in enumerate(imgs):
        try:
            data = img.get_fdata()
        except AttributeError:
            try:
                data = np.ndarray(img)  # input is array with less than 32 dims
            except ValueError:
                data = img  # input is array

        assert isinstance(
            data, np.ndarray), "Input images need to be nifti images or np.ndarrays!"
        assert data.ndim == 3, f"Input images need to be 3D! Data has {data.ndim} dimensions."

        x, y, z = data.nonzero()
        values = data[x, y, z]
        hover_text = [names[idx]] * len(x)

        # Cycle through colors if more images than color scales
        color = colors[idx % len(colors)]

        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=size,
                color=values,
                colorscale=color,
                opacity=opacity
            ),
            text=hover_text,
            hoverinfo='text',
            name=names[idx]
        ))

    fig.update_layout(
        scene={
            'xaxis_title': 'X Axis',
            'yaxis_title': 'Y Axis',
            'zaxis_title': 'Z Axis'
        },
        title='3D Visualization of Segmented Images',
    )

    fig.show()


def explore_scan(array, layer, dim3):
    """
    Visual interaction with medical scan.
    Example call:
        mtx = nib.load(...).get_fdata()
        interact(explore_scan, array=fixed(mtx),
                 layer=(0, min(mtx.shape[0]-1, mtx.shape[1]-1)),
                 dim3=(0, mtx.shape[2]-1)
                 )
    """
    _, axs = plt.subplots(1, 3, figsize=(18, 6))
    panes = ['sagittal', 'coronal', 'transverse/axial']
    mtx = array
    mtx = mtx / mtx.max()  # normalize the matrix
    col = ["YlOrRd", "viridis", "plasma", 'grey'][-2]
    alpha = 0.9
    for dim in range(len(mtx.shape)):
        if dim == 0:
            axs[dim].imshow(mtx[layer, :, :], cmap=col, alpha=alpha)
        elif dim == 1:
            axs[dim].imshow(mtx[:, layer, :], cmap=col, alpha=alpha)
        else:
            axs[dim].imshow(mtx[:, :, dim3], cmap=col, alpha=alpha)

        axs[dim].set_title(
            f"Dimension {dim+1}: {panes[dim]} plane",
            family='Arial',
            fontsize=20)
        axs[dim].axis('off')

    plt.tight_layout()

    return layer, dim3


def nifit_info(img: nib.nifti1.Nifti1Image,
               return_stats: bool = False) -> None:
    """
    Takes in a nifti image (nibabel object) and
    1 prints out the shape of the image along each axis
    2 prints out the voxel sizes
    """
    mtx = img.get_fdata()
    shape = mtx.shape

    affine = img.affine

    # extract the voxel sizes from the diagonal elements of the affine matrix
    voxel_sizes = affine[:3, :3].diagonal()

    if return_stats:
        return shape, voxel_sizes

    # sag, cor, ax = shape
    panes = ['sagittal', 'coronal', 'axial']
    print('=' * 70)
    for i, p in enumerate(panes):
        print(f"{p.ljust(8)}:\t{shape[i]}")
    print("Voxel   :\tx=%.3f, y=%.3f, z=%.3f" % (*voxel_sizes,))
    print('=' * 70)
    return None
