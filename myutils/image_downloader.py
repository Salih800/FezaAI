import logging
import time
import os
import requests


def download_image(img_url, images_folder):
    t1 = time.perf_counter()
    image_name = img_url.split("/")[-1]  # frame_x.jpg

    image_path = images_folder + image_name

    if not os.path.isdir(images_folder):
        os.makedirs(images_folder)

    if not os.path.isfile(images_folder + image_name):
        img_bytes = requests.get(img_url).content

        with open(image_path, 'wb') as img_file:
            img_file.write(img_bytes)

        t2 = time.perf_counter()

        logging.info(f'{image_path} - Download Finished in {round(t2 - t1, 2)} seconds')
        return True

    else:
        logging.info(f"Image was already here: {image_path}")
        return False

