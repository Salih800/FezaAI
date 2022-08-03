import logging
import time

from datetime import timedelta

from src.connection_handler import ConnectionHandler

from myutils.ConnectionInfo import ConnectionInfo
from myutils.logger_setter import set_logger
from myutils.image_downloader import download_image


def run():
    connection_info = ConnectionInfo()

    set_logger(connection_info.team_name)

    server = ConnectionHandler(connection_info.evaluation_server_url,
                               username=connection_info.team_name,
                               password=connection_info.password)

    frame_list = server.get_frames()

    logging.info(f"{len(frame_list)} Total images found.")

    time_start = time.time()
    downloaded_image_count = 0
    already_here_image_count = 0

    for i, frame in enumerate(frame_list):
        img_url = connection_info.evaluation_server_url + "media" + frame['image_url']
        img_save_path = "./_images/" + frame['video_name'] + "/"

        logging.info(f"Looking for {img_url}")

        if download_image(img_url, img_save_path):
            downloaded_image_count += 1
        else:
            already_here_image_count += 1

    logging.info(f"{downloaded_image_count} images downloaded and"
                 f" {already_here_image_count} images already downloaded.")
    logging.info(f"Total time: {timedelta(seconds=time.time()-time_start)}")


if __name__ == '__main__':
    run()


