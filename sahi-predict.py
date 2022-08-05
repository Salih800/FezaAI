import glob
import os
import shutil
from pathlib import Path
import sys

sys.path.append("./sahi")
sys.path.append("./yolov7")
sys.path.append("./yolov5")

from myutils.model_download import download_model

# import required functions, classes
from sahi.model import Yolov5DetectionModel, Yolov7DetectionModel, YoloDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict

from src.our_models import get_model_info, MODELS

models = MODELS()

model = get_model_info(models.yolov5x6_yaya_arac)
download_model(model.gdrive_id, model.path)

detection_model = YoloDetectionModel(
    model_path=model.path,
    confidence_threshold=model.confidence_threshold,
    image_size=model.image_size,
    which_yolo=model.which_yolo
)

images_path = "./images/"
label_save_path = "./labels/"
detected_images_path = "./_detected_sahi_images/"
# Path(images_path).mkdir(parents=True, exist_ok=True)
Path(detected_images_path).mkdir(parents=True, exist_ok=True)
if os.path.isdir(label_save_path):
    shutil.rmtree(label_save_path)
os.makedirs(label_save_path)

image_list = glob.glob(images_path + "*g")
print(f"Total Pictures: {len(image_list)}")
for i, image in enumerate(image_list):
    print(f"{i + 1}/{len(image_list)}", image)
    image_name = os.path.split(image)[-1][:-4]
    image_type = ".jpg"

    result = get_prediction(image, detection_model)
    print(f"{model.name} normal detection time: {result.durations_in_seconds}")
    file_name = f"{image_name}-{model.name}-result"
    result.export_visuals(file_name=file_name,
                          export_dir=detected_images_path)

    result.save_yolo_label(label_path=label_save_path + "normal/" + model.name + "/" + image_name + ".txt")

    slice_512 = 512
    slice_256 = 256
    overlap_ratio = 0.2

    auto_sliced_result = get_sliced_prediction(
        image,
        detection_model,
    )
    file_name = f"{image_name}-{model.name}-sliced-auto-result"
    auto_sliced_result.export_visuals(
        file_name=file_name,
        export_dir=detected_images_path)

    auto_sliced_result.save_yolo_label(label_path=label_save_path + "sahi-auto/"
                                                  + model.name + "/" + image_name + ".txt")

    print(f"{model.name} auto-sliced detection time: {auto_sliced_result.durations_in_seconds}")
