import argparse
import sys
from pathlib import Path

import torch
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    EnsureTyped
)

from kardia.config import load_config
from kardia.training.trainer import Trainer

def main() -> None:
    parser = argparse.ArgumentParser(description="Kardia Training Pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root (must contain imagesTr and labelsTr)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.is_dir():
        print(f"Error: Directory {data_dir} does not exist.")
        sys.exit(1)

    data_dicts = []
    gt_files = list(data_dir.rglob("*_gt.nii.gz"))
    if gt_files:
        print("Detected ACDC dataset structure.")
        for lbl in gt_files:
            img_name = lbl.name.replace("_gt.nii.gz", ".nii.gz")
            img = lbl.parent / img_name
            if img.exists():
                data_dicts.append({"image": str(img), "label": str(lbl)})
    else:
        images_dir = data_dir / "imagesTr"
        labels_dir = data_dir / "labelsTr"
        if images_dir.is_dir() and labels_dir.is_dir():
            print("Detected standard imagesTr/labelsTr structure.")
            images = sorted(list(images_dir.glob("*.nii*")))
            labels = sorted(list(labels_dir.glob("*.nii*")))
            for img in images:
                lbl = labels_dir / img.name
                if lbl.exists():
                    data_dicts.append({"image": str(img), "label": str(lbl)})

    if not data_dicts:
        print("Error: Could not match any images with corresponding labels.")
        sys.exit(1)

    print(f"Found {len(data_dicts)} training samples.")

    config = load_config()
    spatial_size = tuple((int(v) for v in config['model']['spatial_size']))
    prep = config.get('preprocess', {})
    a_min = float(prep.get('a_min', 0.0))
    a_max = float(prep.get('a_max', 2000.0))
    b_min = float(prep.get('b_min', 0.0))
    b_max = float(prep.get('b_max', 1.0))

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True),
        Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("trilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.long))
    ])

    dataset = Dataset(data=data_dicts, transform=transforms)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    trainer = Trainer(config)
    print("Starting training...")
    trainer.train(loader, epochs=args.epochs)
    print("Training complete.")

if __name__ == "__main__":
    main()
