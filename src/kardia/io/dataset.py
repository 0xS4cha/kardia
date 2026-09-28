import urllib.request
import zipfile
import tarfile
import argparse
from pathlib import Path

def download_and_extract(url: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split('/')[-1] or 'dataset.zip'
    filepath = dest_dir / filename
    
    print(f"Downloading {url} to {filepath}...")
    urllib.request.urlretrieve(url, filepath)
    print("Download complete.")
    
    print(f"Extracting {filepath}...")
    if filepath.name.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
    elif filepath.name.endswith('.tar.gz') or filepath.name.endswith('.tgz'):
        with tarfile.open(filepath, 'r:gz') as tar_ref:
            tar_ref.extractall(dest_dir)
    else:
        print("Unknown archive format. Skipping extraction.")
    print(f"Dataset successfully installed in {dest_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Kardia Dataset Installer")
    parser.add_argument("--url", type=str, help="URL of the dataset archive to download", required=True)
    parser.add_argument("--dest", type=str, default="data/dataset", help="Destination folder inside 'data/'")
    args = parser.parse_args()
    
    dest_path = Path.cwd() / args.dest
    download_and_extract(args.url, dest_path)

if __name__ == "__main__":
    main()
