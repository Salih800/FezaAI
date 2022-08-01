import logging
import os.path
import gdown


def download_model(gdrive_id, download_path):
    if not os.path.isdir(download_path):
        logging.warning(f"Model not found in '{download_path}'! Trying to download it...")
        gdown.download(id=gdrive_id, output=download_path, quiet=False)
    else:
        logging.info(f"Model found in {download_path}: {gdown.md5sum(download_path)}")

