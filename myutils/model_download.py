import logging
import os.path
import gdown


def download_model(gdrive_id, download_path):
    if not os.path.isfile(download_path):
        os.makedirs(os.path.split(download_path)[0], exist_ok=True)
        logging.warning(f"Model not found in '{download_path}'! Trying to download it...")
        gdown.download(id=gdrive_id, output=download_path, quiet=False)
    else:
        logging.info(f"Model found in {download_path}: {gdown.md5sum(download_path)}")

