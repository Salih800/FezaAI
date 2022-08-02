import logging
import time

import requests

import imutils
import os
import cv2
from sahi.model import DetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.prediction import PredictionResult


def log_detected_classes(prediction, image_path):
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
    logging.info(f"{image_path} Detected classes: {yaya} yaya, {arac} arac, "
                 f"{uap} uap, {uap_not} uap_not, {uai} uai, {uai_not} uai_not")


class ObjectDetectionModel:
    # Base class for team models

    def __init__(self, evaluation_server_url,
                 uap_uai_model: DetectionModel = None, uap_uai_sliced: bool = False,
                 yaya_arac_model: DetectionModel = None, yaya_arac_sliced: bool = False,
                 view_image: bool = True, save_detected_image: bool = True):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url

        self.images_folder = "./_images/"
        self.detected_images_folder = "./_detected_images/"
        self.detection_results_path = "./_detection_results_txt/"

        self.yaya_arac_model = yaya_arac_model
        self.uap_uai_model = uap_uai_model

        self.uap_uai_sliced = uap_uai_sliced
        self.yaya_arac_sliced = yaya_arac_sliced

        self.view_image = view_image
        self.save_detected_image = save_detected_image

        self.yaya_arac_model.model_name += "-sahi" if self.yaya_arac_sliced else ""
        self.uap_uai_model.model_name += "-sahi" if self.uap_uai_sliced else ""

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
        image_path = self.images_folder[:-1] + prediction.image_url
        save_image_name = os.path.basename(prediction.image_url)[:-4]

        if self.yaya_arac_sliced:
            yaya_arac_result = get_sliced_prediction(image_path, self.yaya_arac_model)
        else:
            yaya_arac_result = get_prediction(image_path, self.yaya_arac_model)

        if self.uap_uai_sliced:
            uap_uai_result = get_sliced_prediction(image_path, self.uap_uai_model)
        else:
            uap_uai_result = get_prediction(image_path, self.uap_uai_model)

        prediction.detected_objects = (yaya_arac_result.to_teknofest_predictions()
                                       + uap_uai_result.to_teknofest_predictions())

        if self.save_detected_image:
            all_detections = []
            for detection in yaya_arac_result.object_prediction_list + uap_uai_result.object_prediction_list:
                if detection.category.name.startswith("u"):
                    detection.category.id += 2
                all_detections.append(detection)
            all_detections = PredictionResult(
                object_prediction_list=all_detections,
                image=yaya_arac_result.image,
            )
            export_image_path = self.detected_images_folder + prediction.video_name + "/" \
                                + self.yaya_arac_model.model_name + "_" + self.uap_uai_model.model_name + "/"
            all_detections.export_visuals(export_dir=export_image_path, file_name=save_image_name)

            if self.view_image:
                cv2.imshow("FRAME",
                           imutils.resize(cv2.imread(export_image_path + save_image_name + ".png"), width=1280))
                cv2.waitKey(1)

            export_label_path = self.detection_results_path + prediction.video_name + "/" \
                                + self.yaya_arac_model.model_name + "_" + self.uap_uai_model.model_name \
                                + "/" + save_image_name + ".txt"

            all_detections.save_yolo_label(export_label_path)

        log_detected_classes(prediction, image_path)

        return prediction
