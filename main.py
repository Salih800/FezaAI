import concurrent.futures
import logging
import time
from datetime import timedelta
from pathlib import Path

from src.connection_handler import ConnectionHandler
from src.frame_predictions import FramePredictions
from src.object_detection_model import ObjectDetectionModel
from src.our_models import get_model_info, MODELS

# import sys
# sys.path.append("./sahi/")
# sys.path.append("./yolov7")
# sys.path.append("./yolov5")

from myutils.model_download import download_model
from myutils.logger_setter import set_logger
from myutils.ConnectionInfo import ConnectionInfo

from sahi.model import YoloDetectionModel, Yolov5DetectionModel, Yolov7DetectionModel


def run():
    print("Started...")
    # Get configurations from .env file
    connection_info = ConnectionInfo()

    # Declare logging configuration.
    # configure_logger(team_name)
    set_logger(connection_info.team_name)

    # Teams can implement their codes within ObjectDetectionModel class. (OPTIONAL)
    models = MODELS()

    yaya_arac_model = get_model_info(models.yolov7_e6e_yaya_arac_v3)
    uap_uai_model = get_model_info(models.yolov7_uap_uai)

    download_model(yaya_arac_model.gdrive_id, yaya_arac_model.path)
    download_model(uap_uai_model.gdrive_id, uap_uai_model.path)

    if yaya_arac_model.which_yolo != uap_uai_model.which_yolo:
        exception = "You can not load two different type model at the same time!"
        logging.error(exception)
        raise Exception(exception)

    yaya_arac_detection_model = YoloDetectionModel(
        model_path=yaya_arac_model.path,
        confidence_threshold=yaya_arac_model.confidence_threshold,
        image_size=yaya_arac_model.image_size,
        model_name=yaya_arac_model.name,
        which_yolo=yaya_arac_model.which_yolo
    )

    uap_uai_detection_model = YoloDetectionModel(
        model_path=uap_uai_model.path,
        confidence_threshold=uap_uai_model.confidence_threshold,
        image_size=uap_uai_model.image_size,
        model_name=uap_uai_model.name,
        which_yolo=uap_uai_model.which_yolo
    )

    logging.info(f"Model infos: \n\t"
                 f"{yaya_arac_detection_model}\n\t"
                 f"{uap_uai_detection_model}\n")

    using_models = yaya_arac_detection_model.model_name + "_" + uap_uai_detection_model.model_name

    detection_model = ObjectDetectionModel(connection_info.evaluation_server_url,
                                           yaya_arac_model=yaya_arac_detection_model,
                                           yaya_arac_sliced=False,
                                           uap_uai_model=uap_uai_detection_model,
                                           uap_uai_sliced=False,
                                           download_again=False,
                                           vehicle_conf=0.8)

    # Connect to the evaluation server.
    server = ConnectionHandler(connection_info.evaluation_server_url,
                               username=connection_info.team_name,
                               password=connection_info.password)

    # Get all frames from current active session.
    frames_json = server.get_frames()

    # Create images folder
    images_folder = "./_images/"
    Path(images_folder).mkdir(parents=True, exist_ok=True)

    # Run object detection model frame by frame.
    detection_start_time = time.time()
    for i, frame in enumerate(frames_json):
        loop_start = time.time()
        logging.info(f"Picture Number: {i + 1}/{len(frames_json)}")
        # Create a prediction object to store frame info and detections
        predictions = FramePredictions(frame['url'], frame['image_url'], frame['video_name'])

        # Run detection model
        predictions = detection_model.process(predictions, connection_info.evaluation_server_url)
        # Send model predictions of this frame to the evaluation server

        server.save_or_upload_prediction(predictions,
                                         model_name=using_models,
                                         save_payload=True,
                                         upload_payload=True)

        logging.info(f"Loop duration: {round(time.time() - loop_start, 2)}")

    logging.info(f"{len(frames_json)} images processed in {timedelta(seconds=time.time() - detection_start_time)}")


if __name__ == '__main__':
    run()
