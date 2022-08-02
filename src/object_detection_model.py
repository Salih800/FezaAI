import datetime
import json
import logging
import time

import requests

from src.our_models import ModelInfo
from src.constants import classes, landing_statuses
from src.detected_object import DetectedObject
import imutils
import torch
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
from sahi.model import DetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.prediction import PredictionResult


class ObjectDetectionModel:
    # Base class for team models

    def __init__(self, evaluation_server_url,
                 uap_uai_model: DetectionModel = None, uap_uai_sliced: bool = False,
                 yaya_arac_model: DetectionModel = None, yaya_arac_sliced: bool = False,
                 view_image: bool = True):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        self.images_folder = "./_images/"

        self.yaya_arac_model = yaya_arac_model
        self.uap_uai_model = uap_uai_model

        self.uap_uai_sliced = uap_uai_sliced
        self.yaya_arac_sliced = yaya_arac_sliced

        self.view_image = view_image

    @staticmethod
    def download_image(img_url, images_folder):
        t1 = time.perf_counter()
        img_bytes = requests.get(img_url).content
        image_name = img_url.split("/")[-1]  # frame_x.jpg

        if not os.path.isdir(images_folder):
            os.makedirs(images_folder)

        with open(images_folder + image_name, 'wb') as img_file:
            img_file.write(img_bytes)

        t2 = time.perf_counter()

        logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')

    def process(self, prediction, evaluation_server_url):
        # Yarışmacılar resim indirme, pre ve post process vb işlemlerini burada gerçekleştirebilir.
        # Download image (Example)
        time_start = time.time()
        self.download_image(evaluation_server_url + "media" + prediction.image_url, "./_images"
                            + os.path.dirname(prediction.image_url) + "/")
        logging.info(f"Download seconds: {round(time.time() - time_start, 2)}")
        # Örnek: Burada OpenCV gibi bir tool ile preprocessing işlemi yapılabilir. (Tercihe Bağlı)
        # ...
        # Nesne tespiti modelinin bulunduğu fonksiyonun (self.detect() ) çağırılması burada olmalıdır.
        frame_results = self.detect(prediction)
        # Tahminler objesi FramePrediction sınıfında return olarak dönülmelidir.
        return frame_results

    def detect(self, prediction):
        detection_results_path = "./_detection_results/" + prediction.video_name + "/"
        if not os.path.isdir(detection_results_path):
            os.makedirs(detection_results_path)

        image_path = self.images_folder[:-1] + prediction.image_url
        save_image_name = os.path.basename(prediction.image_url)[:-4]
        # time_start = time.time()
        if self.yaya_arac_sliced:
            yaya_arac_result = get_sliced_prediction(image_path, self.yaya_arac_model)
        else:
            yaya_arac_result = get_prediction(image_path, self.yaya_arac_model)

        # yaya_arac_result.export_visuals(export_dir="extracted_results/", file_name=save_image_name + "_result")
        # logging.info(f"yaya-arac result seconds: {round(time.time() - time_start, 2)}")
        # time_start = time.time()
        # uap_uai_result = get_prediction(image_path, self.uap_uai_model)
        if self.yaya_arac_sliced:
            uap_uai_result = get_sliced_prediction(image_path, self.uap_uai_model)
        else:
            uap_uai_result = get_prediction(image_path, self.uap_uai_model)
        # logging.info(f"uap-uai result seconds: {round(time.time() - time_start, 2)}")

        prediction.detected_objects = yaya_arac_result.to_teknofest_predictions() + uap_uai_result.to_teknofest_predictions()

        all_detections = PredictionResult(
            object_prediction_list=yaya_arac_result.object_prediction_list + uap_uai_result.object_prediction_list,
            image=yaya_arac_result.image,
        )

        all_detections.export_visuals(export_dir="extracted_results/", file_name=save_image_name + "_result")
        if self.view_image:
            cv2.imshow("img", imutils.resize(cv2.imread("extracted_results/" + save_image_name + "_result.png"), width=1280))
            cv2.waitKey(1)

        detections_filename = datetime.datetime.now().strftime(detection_results_path +
                                                               os.path.basename(prediction.image_url)[:-4]
                                                               + '__%Y_%m_%d__%H_%M_%S_%f.txt')

        yaya_arac_result.save_yolo_label(detections_filename)
        uap_uai_result.save_yolo_label(detections_filename)
        # with open(detections_filename, "w") as all_json_file:
        #     json.dump(all_json_results, all_json_file)
        arac, yaya, uap, uap_not, uai, uai_not = 0, 0, 0, 0, 0, 0
        for detection in prediction.detected_objects:
            if detection.cls == 0:
                arac += 1
            elif detection.cls == 1:
                yaya += 1
            elif detection.cls == 2:
                if detection.landing_status == 1:
                    uap += 1
                else:
                    uap_not += 1
            elif detection.cls == 3:
                if detection.landing_status == 1:
                    uai += 1
                else:
                    uai_not += 1
        print(f"Detected classes: {yaya} yaya, {arac} arac, {uap} uap, {uap_not} uap_not, {uai} uai, {uai_not} uai_not")

        return prediction
