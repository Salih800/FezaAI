import concurrent.futures
import logging
from datetime import datetime
from pathlib import Path

from decouple import config

from src.connection_handler import ConnectionHandler
from src.frame_predictions import FramePredictions
from src.object_detection_model import ObjectDetectionModel


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
    configure_logger(team_name)

    # Teams can implement their codes within ObjectDetectionModel class. (OPTIONAL)
    detection_model = ObjectDetectionModel(evaluation_server_url)

    # Connect to the evaluation server.
    server = ConnectionHandler(evaluation_server_url, username=team_name, password=password)

    # Get all frames from current active session.
    frames_json = server.get_frames()

    # Create images folder
    images_folder = "./_images/"
    Path(images_folder).mkdir(parents=True, exist_ok=True)

    # Run object detection model frame by frame.
    for frame in frames_json:
        # Create a prediction object to store frame info and detections
        predictions = FramePredictions(frame['url'], frame['image_url'], frame['video_name'])
        #print(predictions.image_url)
        # Run detection model
        predictions = detection_model.process(predictions,evaluation_server_url)
        # Send model predictions of this frame to the evaluation server
        result = server.send_prediction(predictions)


if __name__ == '__main__':
    run()
