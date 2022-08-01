import glob
import os
import time
from pathlib import Path
import sys
# print(os.getcwd())
sys.path.append("./sahi")
sys.path.append("./yolov7")
sys.path.append("./yolov5")
# print(sys.path)
from sahi.utils.yolov5 import (
    download_yolov5s6_model, download_yolov5l6_model
)

from myutils.model_download import download_model

# import required functions, classes
from sahi.model import Yolov5DetectionModel, Yolov7DetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
# from IPython.display import Image
from PIL import Image
from src.our_models import get_model_info, MODELS
import sys

# sys.path.append("../")
# print(sys.path)
# exit()
# yolov5_model_path = '../models/yolov5s6-b8-e300-i1920-vismix+teknofest.pt'
# yolov5_model_path = "models/yolov5l6.pt"
# download_yolov5s6_model(destination_path=yolov5_model_path)

# download test images into demo_data folder
# download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg',
#                   'demo_data/small-vehicles1.jpeg')
# download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png',
#                   'demo_data/terrain2.png')

models = MODELS()
# yaya_arac_model = get_model_info(models.yolov7_e6e_yaya_arac)
model = get_model_info(models.yolov7_e6e_yaya_arac)
download_model(model.gdrive_id, model.path)

# download_yolov5s6_model(destination_path=model.path)

# detection_model = Yolov5DetectionModel(
#     model_path=model.sahi_path,
#     confidence_threshold=0.3,
#     image_size=model.size,
# )

detection_model = Yolov7DetectionModel(
    model_path=model.path,
    confidence_threshold=model.conf,
    image_size=model.size
)

images_path = "../images/"
detected_images_path = "sahi/_detected_images/"
# Path(images_path).mkdir(parents=True, exist_ok=True)
Path(detected_images_path).mkdir(parents=True, exist_ok=True)

# print(detection_model.num_categories, detection_model.category_names)
image_list = glob.glob(images_path + "*g")
print(f"Total Pictures: {len(image_list)}")
for i, image in enumerate(image_list[:2]+image_list[5:9]):
    print(f"{i+1}/{len(image_list)}", image)
    image_name = os.path.split(image)[-1][:-4]
    image_type = ".jpg"

    result = get_prediction(image, detection_model)
    print(f"{model.name} normal detection time: {result.durations_in_seconds}")

    result.export_visuals(file_name=f"{image_name}-{model.name}-result",
                          export_dir=detected_images_path)

    slice_512 = 512
    slice_256 = 256
    overlap_ratio = 0.2

    sliced_result_256 = get_sliced_prediction(
        image,
        detection_model,
        slice_height=slice_256,
        slice_width=slice_256,
        overlap_width_ratio=overlap_ratio,
        overlap_height_ratio=overlap_ratio,
    )

    print(f"{model.name} {slice_256}x{slice_256} - {overlap_ratio} sliced detection time: {sliced_result_256.durations_in_seconds}")

    sliced_result_256.export_visuals(
        file_name=f"{image_name}-{model.name}-sliced-{slice_256}x{slice_256}-{overlap_ratio}-result",
        export_dir=detected_images_path)

    sliced_result_512 = get_sliced_prediction(
        image,
        detection_model,
        slice_height=slice_512,
        slice_width=slice_512,
        overlap_width_ratio=overlap_ratio,
        overlap_height_ratio=overlap_ratio,
    )

    print(f"{model.name} {slice_512}x{slice_512} - {overlap_ratio} sliced detection time: {sliced_result_512.durations_in_seconds}")

    sliced_result_512.export_visuals(
        file_name=f"{image_name}-{model.name}-sliced-{slice_512}x{slice_512}-{overlap_ratio}-result",
        export_dir=detected_images_path)

    auto_sliced_result = get_sliced_prediction(
        image,
        detection_model,
        # slice_height=slice_height,
        # slice_width=slice_width,
        # overlap_width_ratio=overlap_ratio,
        # overlap_height_ratio=overlap_ratio,
    )

    auto_sliced_result.export_visuals(
        file_name=f"{image_name}-{model.name}-sliced-auto-result",
        export_dir=detected_images_path)

    print(f"{model.name} auto-sliced detection time: {auto_sliced_result.durations_in_seconds}")

    # slice_height = slice_size,
    # slice_width = slice_size,
    # overlap_height_ratio = overlap_ratio,
    # overlap_width_ratio = overlap_ratio

    # for p in result.object_prediction_list:
    #     print(f"Category Id: {p.category.id}, Category Name: {p.category.name}, Confidence: {round(p.score.value, 3)}, "
    #           f"BBOX: {p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy}")


# Image.open("demo_data/prediction_visual.png").show(title="demo_data/prediction_visual.png")
