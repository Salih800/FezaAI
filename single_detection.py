import concurrent.futures
import logging
import time
from datetime import timedelta
from pathlib import Path

from src.connection_handler import ConnectionHandler
from src.frame_predictions import FramePredictions
from src.single_detection_model import SingleDetectionModel
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

    model = get_model_info(models.yolov5s_yaya_arac)

    download_model(model.gdrive_id, model.path)

    detection_model = YoloDetectionModel(
        model_path=model.path,
        confidence_threshold=model.confidence_threshold,
        image_size=model.image_size,
        model_name=model.name,
        which_yolo=model.which_yolo
    )

    logging.info(f"Model info: \n\t"
                 f"{detection_model}\n")

    detection_model = SingleDetectionModel(connection_info.evaluation_server_url,
                                           model=detection_model,
                                           model_sliced=False,
                                           download_again=False)

    # Connect to the evaluation server.
    server = ConnectionHandler(connection_info.evaluation_server_url,
                               username=connection_info.team_name,
                               password=connection_info.password)

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
        logging.info(f"Picture Number: {i + 1}/{len(frames_json)}")
        # Create a prediction object to store frame info and detections
        predictions = FramePredictions(frame['url'], frame['image_url'], frame['video_name'])

        # Run detection model
        predictions = detection_model.process(predictions, connection_info.evaluation_server_url)
        # Send model predictions of this frame to the evaluation server

        server.save_or_upload_prediction(predictions,
                                         model_name=model.name,
                                         save_payload=True,
                                         upload_payload=True)

        logging.info(f"Loop duration: {round(time.time() - loop_start, 2)}")

    logging.info(f"{len(frames_json)} images processed in {timedelta(seconds=time.time() - detection_start_time)}")


if __name__ == '__main__':
    run()
