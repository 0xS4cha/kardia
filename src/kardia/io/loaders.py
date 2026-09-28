from __future__ import annotations
from pathlib import Path
import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk
from .volume import Volume

def load_volume(path: str | Path) -> Volume:
    target = Path(path)
    if target.is_dir():
        return load_dicom_series(target)
    suffix = ''.join(target.suffixes).lower()
    if suffix in {'.nii', '.nii.gz'} or target.suffix.lower() in {'.nrrd', '.mha', '.mhd'}:
        return load_nifti(target)
    if target.suffix.lower() == '.dcm':
        return load_dicom_series(target.parent)
    return load_nifti(target)

def load_nifti(path: str | Path) -> Volume:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f'Fichier introuvable : {target}')
    try:
        image = sitk.ReadImage(str(target))
        return Volume.from_sitk(image, path=target)
    except Exception:
        return _load_nifti_nibabel(target)

def _load_nifti_nibabel(path: Path) -> Volume:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim > 3:
        data = np.take(data, 0, axis=-1)
    array = np.transpose(np.ascontiguousarray(data), (2, 1, 0)).astype(np.float32)
    zooms = img.header.get_zooms()[:3]
    affine = img.affine
    origin = (float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3]))
    return Volume(array=array, spacing=(float(zooms[0]), float(zooms[1]), float(zooms[2])), origin=origin, direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), path=path)

def load_dicom_series(folder: str | Path) -> Volume:
    directory = Path(folder)
    if not directory.is_dir():
        raise NotADirectoryError(f'Dossier DICOM invalide : {directory}')
    _assert_contains_dicom(directory)
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(directory))
    if not series_ids:
        raise ValueError(f'Aucune série DICOM dans {directory}')
    best_id = max(series_ids, key=lambda sid: len(reader.GetGDCMSeriesFileNames(str(directory), sid)))
    filenames = reader.GetGDCMSeriesFileNames(str(directory), best_id)
    reader.SetFileNames(filenames)
    image = reader.Execute()
    return Volume.from_sitk(image, path=directory)

def _assert_contains_dicom(directory: Path) -> None:
    for candidate in directory.iterdir():
        if not candidate.is_file():
            continue
        try:
            pydicom.dcmread(str(candidate), stop_before_pixels=True, force=False)
            return
        except Exception:
            if candidate.suffix.lower() == '.dcm':
                return
    raise ValueError(f'Aucun fichier DICOM lisible dans {directory}')