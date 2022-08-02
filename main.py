import concurrent.futures
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

from decouple import config

from src.connection_handler import ConnectionHandler
from src.frame_predictions import FramePredictions
from src.object_detection_model import ObjectDetectionModel
from src.our_models import get_model_info, MODELS

from myutils.model_download import download_model

from sahi.model import Yolov5DetectionModel, Yolov7DetectionModel


def set_logger(team_name):
    log_folder = "./_logs/"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime(log_folder + team_name + '_%Y_%m_%d__%H_%M_%S_%f.log')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_filename)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


def configure_logger(team_name):
    log_folder = "./_logs/"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime(log_folder + team_name + '_%Y_%m_%d__%H_%M_%S_%f.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def run():
    print("Started...")
    # Get configurations from .env file
    config.search_path = "./config/"
    team_name = config('TEAM_NAME')
    password = config('PASSWORD')
    evaluation_server_url = config("EVALUATION_SERVER_URL")

    # Declare logging configuration.
    # configure_logger(team_name)
    set_logger(team_name)

    # Teams can implement their codes within ObjectDetectionModel class. (OPTIONAL)
    models = MODELS()
    yaya_arac_model = get_model_info(models.yolov5x6_yaya_arac)
    uap_uai_model = get_model_info(models.yolov5s_uap_uai)
    download_model(yaya_arac_model.gdrive_id, yaya_arac_model.path)
    download_model(uap_uai_model.gdrive_id, uap_uai_model.path)

    yaya_arac_detection_model = Yolov5DetectionModel(
        model_path=yaya_arac_model.path,
        confidence_threshold=yaya_arac_model.conf,
        image_size=yaya_arac_model.size,
    )

    uap_uai_detection_model = Yolov5DetectionModel(
        model_path=uap_uai_model.path,
        confidence_threshold=uap_uai_model.conf,
        image_size=uap_uai_model.size,
    )

    detection_model = ObjectDetectionModel(evaluation_server_url,
                                           yaya_arac_model=yaya_arac_detection_model,
                                           yaya_arac_sliced=False,
                                           uap_uai_model=uap_uai_detection_model,
                                           uap_uai_sliced=False)

    # Connect to the evaluation server.
    server = ConnectionHandler(evaluation_server_url, username=team_name, password=password)

    # Get all frames from current active session.
    frames_json = server.get_frames()

    # Create images folder
    images_folder = "./_images/"
    Path(images_folder).mkdir(parents=True, exist_ok=True)

    prediction_sent = False
    # Run object detection model frame by frame.
    detection_start_time = time.time()
    for i, frame in enumerate(frames_json):
        loop_start = time.time()
        logging.info(f"Picture Number: {i+1}/{len(frames_json)}")
        # Create a prediction object to store frame info and detections
        predictions = FramePredictions(frame['url'], frame['image_url'], frame['video_name'])

        # Run detection model
        predictions = detection_model.process(predictions, evaluation_server_url)
        # Send model predictions of this frame to the evaluation server
        while not prediction_sent:
            result = server.save_or_upload_prediction(predictions,
                                                      model_name=yaya_arac_model.name + "_" + uap_uai_model.name,
                                                      save_payload=True, upload_payload=True)
            if result:
                if result.status_code == 201:
                    prediction_sent = True
                elif result.status_code == 406:
                    prediction_sent = True
                else:
                    print("Sleeping 10 seconds")
                    time.sleep(10)
            else:
                prediction_sent = True
        prediction_sent = False

        logging.info(f"Loop seconds: {round(time.time() - loop_start, 2)}")

    logging.info(f"{len(frames_json)} images processed in {timedelta(seconds=time.time() - detection_start_time)}")


if __name__ == '__main__':
    run()
