import os
import torch
import numpy as np
import SimpleITK as sitk
from scipy.spatial import KDTree


def map_skeleton_to_volume(
    skeleton_coords,
    node_labels,
    seg_path,
    output_path,
    lung_mask_path=None,
):
    """
    Expands sparse graph skeleton predictions into a dense 3D NIfTI vessel segmentation volume
    using KDTree nearest-neighbor lookup, strictly filtered by the lung parenchyma mask.

    Args:
        skeleton_coords (Tensor or np.ndarray): Graph node coordinates [N, 3] (x, y, z).
        node_labels (Tensor or np.ndarray): Binary node predictions [N] (0: Artery, 1: Vein).
        seg_path (str or Path): Path to reference binary vessel segmentation volume.
        output_path (str or Path): Destination path for mapped 3D NIfTI (.nii.gz).
        lung_mask_path (str or Path, optional): Path to lung binary mask to filter non-parenchymal voxels.

    Returns:
        sitk.Image: SimpleITK image of the mapped volume.
    """
    mask_itk = sitk.ReadImage(str(seg_path))
    mask_array = sitk.GetArrayFromImage(mask_itk)

    # Filter out vessel voxels outside lung parenchyma (e.g., heart, mediastinum)
    if lung_mask_path and os.path.isfile(str(lung_mask_path)):
        lung_mask_itk = sitk.ReadImage(str(lung_mask_path))
        lung_mask_array = sitk.GetArrayFromImage(lung_mask_itk)
        valid_voxels = (mask_array > 0) & (lung_mask_array > 0)
    else:
        valid_voxels = (mask_array > 0)

    z_idx, y_idx, x_idx = np.where(valid_voxels)
    voxel_coords = np.stack([x_idx, y_idx, z_idx], axis=-1)

    if torch.is_tensor(skeleton_coords):
        coords_np = skeleton_coords.detach().cpu().numpy()
    else:
        coords_np = np.asarray(skeleton_coords)

    if torch.is_tensor(node_labels):
        labels_np = node_labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(node_labels)

    tree = KDTree(coords_np)
    _, nearest_indices = tree.query(voxel_coords)
    # Output labels: 0=background, 1=artery, 2=vein
    mapped_labels = labels_np[nearest_indices] + 1

    output_array = np.zeros_like(mask_array, dtype=np.uint8)
    output_array[z_idx, y_idx, x_idx] = mapped_labels
    
    out_itk = sitk.GetImageFromArray(output_array)
    out_itk.CopyInformation(mask_itk)
    
    os.makedirs(os.path.dirname(os.path.abspath(str(output_path))), exist_ok=True)
    sitk.WriteImage(out_itk, str(output_path))
    return out_itk
