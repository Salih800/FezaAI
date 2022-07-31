import urllib.request
from os import path
from pathlib import Path
from typing import Optional


class Yolov5TestConstants:
    YOLOV5N_MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt"
    YOLOV5N_MODEL_PATH = "tests/data/models/yolov5/yolov5n.pt"

    YOLOV5S6_MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s6.pt"
    YOLOV5S6_MODEL_PATH = "tests/data/models/yolov5/yolov5s6.pt"

    YOLOV5M6_MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m6.pt"
    YOLOV5M6_MODEL_PATH = "tests/data/models/yolov5/yolov5m6.pt"

    YOLOV5L6_MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5l6.pt"
    YOLOV5L6_MODEL_PATH = "tests/data/models/yolov5/yolov5l6.pt"

    YOLOV5X6_MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5x6.pt"
    YOLOV5X6_MODEL_PATH = "tests/data/models/yolov5/yolov5x6.pt"


def download_yolov5n_model(destination_path: Optional[str] = None):

    if destination_path is None:
        destination_path = Yolov5TestConstants.YOLOV5N_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov5TestConstants.YOLOV5N_MODEL_URL,
            destination_path,
        )


def download_yolov5s6_model(destination_path: Optional[str] = None):

    if destination_path is None:
        destination_path = Yolov5TestConstants.YOLOV5S6_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov5TestConstants.YOLOV5S6_MODEL_URL,
            destination_path,
        )


def download_yolov5m6_model(destination_path: Optional[str] = None):

    if destination_path is None:
        destination_path = Yolov5TestConstants.YOLOV5M6_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov5TestConstants.YOLOV5M6_MODEL_URL,
            destination_path,
        )


def download_yolov5l6_model(destination_path: Optional[str] = None):

    if destination_path is None:
        destination_path = Yolov5TestConstants.YOLOV5L6_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov5TestConstants.YOLOV5L6_MODEL_URL,
            destination_path,
        )


def download_yolov5x6_model(destination_path: Optional[str] = None):

    if destination_path is None:
        destination_path = Yolov5TestConstants.YOLOV5X6_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov5TestConstants.YOLOV5X6_MODEL_URL,
            destination_path,
        )
