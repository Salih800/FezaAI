import datetime
import json
import logging
import time

import requests

from src.constants import classes, landing_statuses
from src.detected_object import DetectedObject

import torch
import os
from PIL import Image, ImageDraw, ImageFont
import PIL


def draw_bounding_boxes(image_path, json_result_list):
    detected_images_folder = "./_detected_images/"
    if not os.path.isdir(detected_images_folder):
        os.mkdir(detected_images_folder)

    font = ImageFont.truetype(r'arial.ttf', 20)
    img = Image.open(image_path).convert("RGBA")
    img_shape = img.size
    draw = ImageDraw.Draw(img)
    for json_result in json_result_list:
        label = f"{json_result['name']} {round(json_result['confidence'], 2)}"
        w, h = font.getsize(label)
        draw.rectangle((json_result["xmin"], json_result["ymin"],
                        min(img_shape[0] - 1, json_result["xmax"]),
                        min(img_shape[1] - 1, json_result["ymax"])),
                       outline=(255, 0, 0), width=2)

        draw.rectangle((json_result["xmin"], json_result["ymin"],
                        json_result["xmin"] + w, json_result["ymin"] - h),
                       outline=(255, 0, 0), width=2, fill=(255, 0, 0))

        draw.text((json_result["xmin"], json_result["ymin"] - h), label, font=font)
    img.convert("RGB").save(detected_images_folder + os.path.basename(image_path))


class ObjectDetectionModel:
    # Base class for team models

    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        self.images_folder = "./_images/"

        self.start_time = time.time()
        self.detected_pictures = 0
        self.main_model_name = "visdrone_model_v2.pt"
        self.main_model_size = 800
        self.main_model = torch.hub.load("./yolov5", 'custom', source='local', path="models/" + self.main_model_name)
        self.shape_model_name = "uap_uai-v2.pt"
        self.shape_model_size = 640
        self.shape_model = torch.hub.load("./yolov5", 'custom', source='local', path="models/" + self.shape_model_name)
        # Modelinizi bu kısımda init edebilirsiniz.
        # self.model = get_keras_model() # Örnektir!

    @staticmethod
    def download_image(img_url, images_folder):
        t1 = time.perf_counter()
        img_bytes = requests.get(img_url).content
        image_name = img_url.split("/")[-1]  # frame_x.jpg

        with open(images_folder + image_name, 'wb') as img_file:
            img_file.write(img_bytes)

        t2 = time.perf_counter()

        logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')

    def process(self, prediction, evaluation_server_url):
        # Yarışmacılar resim indirme, pre ve post process vb işlemlerini burada gerçekleştirebilir.
        # Download image (Example)
        self.download_image(evaluation_server_url + "media" + prediction.image_url, "./_images/")
        # Örnek: Burada OpenCV gibi bir tool ile preprocessing işlemi yapılabilir. (Tercihe Bağlı)
        # ...
        # Nesne tespiti modelinin bulunduğu fonksiyonun (self.detect() ) çağırılması burada olmalıdır.
        frame_results = self.detect(prediction)
        # Tahminler objesi FramePrediction sınıfında return olarak dönülmelidir.
        return frame_results

    def detect(self, prediction):
        detection_results_path = "./_detection_results/"
        if not os.path.isdir(detection_results_path):
            os.mkdir(detection_results_path)
        if self.detected_pictures >= 79:
            logging.warning("You have sent 79 pictures! Waiting...")
            while time.time() - self.start_time <= 60:
                time.sleep(1)
            self.start_time = time.time()
            self.detected_pictures = 0

        # Modelinizle bu fonksiyon içerisinde tahmin yapınız.
        # results = self.model.evaluate(...) # Örnektir.
        image_path = os.path.join(self.images_folder, os.path.basename(prediction.image_url))
        main_results = self.main_model(image_path, self.main_model_size)
        shape_results = self.shape_model(image_path, self.shape_model_size)

        all_json_results = []

        main_json_results = json.loads(main_results.pandas().xyxy[0].to_json(orient="records"))

        for json_result in main_json_results:
            all_json_results.append(json_result)
            confidence = json_result["confidence"]
            cls = classes[json_result["name"]]
            landing_status = landing_statuses["Inis Alani Degil"]
            top_left_x = json_result["xmin"]
            top_left_y = json_result["ymin"]
            bottom_right_x = json_result["xmax"]
            bottom_right_y = json_result["ymax"]

            d_obj = DetectedObject(cls,
                                   landing_status,
                                   top_left_x,
                                   top_left_y,
                                   bottom_right_x,
                                   bottom_right_y)
            prediction.add_detected_object(d_obj)

        shape_json_results = json.loads(shape_results.pandas().xyxy[0].to_json(orient="records"))

        for json_result in shape_json_results:
            all_json_results.append(json_result)
            confidence = json_result["confidence"]
            cls = classes[json_result["name"]]
            if cls in ["uap", "uai"]:
                landing_status = landing_statuses["Inilebilir"]
            else:
                landing_status = landing_statuses["Inis Alani Degil"]
            top_left_x = json_result["xmin"]
            top_left_y = json_result["ymin"]
            bottom_right_x = json_result["xmax"]
            bottom_right_y = json_result["ymax"]

            d_obj = DetectedObject(cls,
                                   landing_status,
                                   top_left_x,
                                   top_left_y,
                                   bottom_right_x,
                                   bottom_right_y)
            prediction.add_detected_object(d_obj)

        draw_bounding_boxes(image_path, all_json_results)

        payload_filename = datetime.datetime.now().strftime(detection_results_path +
                                                            os.path.basename(prediction.image_url)[:-4]
                                                            + '__%Y_%m_%d__%H_%M_%S_%f.json')

        with open(payload_filename, "w") as all_json_file:
            json.dump(all_json_results, all_json_file)

        self.detected_pictures += 1
        return prediction
