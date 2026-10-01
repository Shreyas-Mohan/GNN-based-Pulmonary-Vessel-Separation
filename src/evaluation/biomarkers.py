import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt


def calculate_copd_biomarkers(img_path, lung_mask_path, vessel_pred_path):
    """
    Computes COPD imaging biomarkers from 3D CT and artery-vein predictions:
    1. Total Lung Volume (mL)
    2. Total Vessel Volume (mL)
    3. Emphysema Index LAA-950 (%) [low attenuation areas < -950 HU]
    4. Artery-to-Vein Ratio (AVR)
    5. Small Vessel Blood Volume Percentage BV5 (%) [radius < 1.26mm]

    Args:
        img_path (str or Path): Path to thoracic CT volume (.nii.gz)
        lung_mask_path (str or Path): Path to lung binary mask (.nii.gz)
        vessel_pred_path (str or Path): Path to vessel prediction volume (1: artery, 2: vein)

    Returns:
        dict: Clinical biomarker measurements with units.
    """
    img_itk = sitk.ReadImage(str(img_path))
    img_arr = sitk.GetArrayFromImage(img_itk)
    lung_mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(lung_mask_path)))
    vessel_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(vessel_pred_path)))
    
    spacing = img_itk.GetSpacing()
    voxel_vol_ml = np.prod(spacing) / 1000.0

    lung_voxels = img_arr[lung_mask_arr > 0]
    total_lung_vol = len(lung_voxels) * voxel_vol_ml
    laa_950 = (
        np.sum(lung_voxels < -950) / len(lung_voxels) * 100.0
        if len(lung_voxels) > 0 else 0.0
    )

    # Restrict vessel volumetry strictly to voxels within lung parenchyma
    in_lung_vessels = vessel_arr * (lung_mask_arr > 0)
    artery_vol = np.sum(in_lung_vessels == 1) * voxel_vol_ml
    vein_vol = np.sum(in_lung_vessels == 2) * voxel_vol_ml
    total_vessel_vol = artery_vol + vein_vol
    avr = artery_vol / vein_vol if vein_vol > 0 else 0.0

    binary_vessels = (in_lung_vessels > 0).astype(np.uint8)
    radius_map = distance_transform_edt(binary_vessels, sampling=spacing[::-1])
    bv5_vol = np.sum((binary_vessels > 0) & (radius_map < 1.26)) * voxel_vol_ml
    bv5_percentage = bv5_vol / total_vessel_vol * 100.0 if total_vessel_vol > 0 else 0.0

    return {
        "Total Lung Volume (mL)": round(float(total_lung_vol), 2),
        "Total Vessel Volume (mL)": round(float(total_vessel_vol), 2),
        "Emphysema Index LAA-950 (%)": round(float(laa_950), 2),
        "Artery-to-Vein Ratio (AVR)": round(float(avr), 4),
        "Small Vessel BV5 (%)": round(float(bv5_percentage), 2),
    }
