from datetime import timedelta, datetime
import glob
import json
import logging
import os.path
import time

import requests


class ConnectionHandler:
    def __init__(self, base_url="http://teknofest.cezerirobot.com:2052/", username=None, password=None):
        self.base_url = base_url
        self.auth_token = None
        self.classes = None
        self.frames = None

        # Define URLs
        self.url_login = self.base_url + "auth/"
        self.url_frames = self.base_url + "frames/"
        self.url_prediction = self.base_url + "prediction/"

        if username and password:
            self.login(username, password)

    def login(self, username, password):
        payload = {'username': username,
                   'password': password}
        files = []
        response = requests.request("POST", self.url_login, data=payload, files=files, timeout=3)
        response_json = json.loads(response.text)
        if response.status_code == 200:
            self.auth_token = response_json['token']
            logging.info("Login Successfully Completed : {}".format(payload))
        else:
            logging.info("Login Failed : {}".format(response.text))

    def get_frames(self):
        """
        Dikkat: Bir dakika içerisinde bir takım maksimum 5 adet get_frames isteği atabilmektedir.
        Bu kısıt yarışma esnasında yarışmacıların gereksiz istek atarak sunucuya yük binmesini
        engellemek için tanımlanmıştır. get_frames fonsiyonunu kullanırken bu kısıtı göz önünde
        bulundurmak yarışmacıların sorumluluğundadır.
        """
        payload = {}
        headers = {
            'Authorization': 'Token {}'.format(self.auth_token)
        }
        images_json_path = "./_images_json_list/"
        if not os.path.isdir(images_json_path):
            os.mkdir(images_json_path)

        response = requests.request("GET", self.url_frames, headers=headers, data=payload)
        self.frames = json.loads(response.text)

        if response.status_code == 200:
            images_json_filename = datetime.now().strftime(images_json_path + self.frames[0]["video_name"]
                                                           + '__%Y_%m_%d__%H_%M_%S_%f.json')
            with open(images_json_filename, "w") as json_file:
                json.dump(self.frames, json_file)
            logging.info("Successful : get_frames : {} pictures saved in the file: {}".format(len(self.frames),
                                                                                              images_json_filename))
        else:
            logging.info("Failed : get_frames : {}".format(response.text))

        return self.frames

    def upload_payload(self, payload):

        files = []
        headers = {
            'Authorization': 'Token {}'.format(self.auth_token),
            'Content-Type': 'application/json',
        }

        waiting_time_limit = 61
        waiting_for_limit = False
        waiting_start_time = 0
        failed_count = 0

        upload_start_time = time.time()

        while True:

            response = requests.request("POST", self.url_prediction, headers=headers, data=payload, files=files)

            if response.status_code == 201:
                upload_end_time = round(time.time() - upload_start_time, 2)
                logging.info(f"Payload uploaded successfully in {upload_end_time}")
                return 1, 0, upload_end_time

            elif response.status_code == 406:
                upload_end_time = round(time.time() - upload_start_time, 2)
                logging.info(f"Payload already uploaded in {upload_end_time}")
                return 0, 1, upload_end_time

            elif response.status_code == 403:
                if not waiting_for_limit:
                    waiting_for_limit = True
                    waiting_start_time = time.time()
                    logging.warning("Limit exceeded. 80frames/min \n\t{}".format(response.text))
                    logging.info("Waiting for limit...")
                elif time.time() - waiting_start_time > waiting_time_limit:
                    raise Exception("Waiting is taking too long!")
                time.sleep(5)

            else:
                failed_count += 1
                if failed_count >= 5:
                    logging.error("Payload send failed 5 times. Ending the upload!")
                    raise Exception("Payload send failed 5 times. Ending the upload!")
                logging.warning("Payload send failed. {}\n\t{}".format(response.status_code, response.text))
                time.sleep(3)

    def save_or_upload_prediction(self, prediction, model_name, upload_payload: bool, save_payload: bool):
        """
        Dikkat: Bir dakika içerisinde bir takım maksimum 80 frame için tahmin gönderebilecektir.
        Bu kısıt yarışma esnasında yarışmacıların gereksiz istek atarak sunucuya yük binmesini
        engellemek için tanımlanmıştır. send_prediction fonsiyonunu kullanırken bu kısıtı göz
        önünde bulundurmak yarışmacıların sorumluluğundadır.

        Öneri: Bir dakika içerisinde gönderilen istek sayısı tutularak sistem hızlı çalışıyorsa
        bekletilebilir (wait() vb). Azami istek sınırı aşıldığında sunucu gönderilen tahmini
        veritabanına yazmamaktadır. Dolayısı ile bu durumu gözardı eden takımların istek sınır
        aşımı yapan gönderimleri değerlendirilMEyecektir. İstek sınırı aşıldığında sunucu aşağıdaki
        cevabı dönmektedir:
        	{"detail":"You do not havle permission to perform this action."}
        Ayrıca yarışmacılar sunucudan bu gibi başarısız bir gönderimi işaret eden cevap alındığında
        gönderilemeyen tahmini sunucuya tekrar göndermek üzere bir mekanizma tasarlayabilir.
        """

        payload = json.dumps(prediction.create_payload(self.base_url))

        if save_payload:
            payloads_path = "./_payloads/" + prediction.video_name + "/" + model_name + "/"
            if not os.path.isdir(payloads_path):
                os.makedirs(payloads_path)

            payload_filename = payloads_path + os.path.basename(prediction.image_url)[:-4] + ".json"

            with open(payload_filename, "w") as detection_file:
                detection_file.write(payload)
            logging.info(f"Payload file saved: {payload_filename}")

        if upload_payload:
            self.upload_payload(payload)

    def upload_payloads(self, payload_folder: str):
        if payload_folder.startswith('"'):
            payload_folder = payload_folder.strip('"')
        if not os.path.isdir(payload_folder):
            raise Exception("Folder path not found!")

        payload_list = glob.glob(payload_folder + "/*.json")
        total_payload_count = len(payload_list)

        logging.info(f"{total_payload_count} Found Total Payloads in '{payload_folder}'")

        uploaded_count = 0
        already_uploaded_count = 0
        total_upload_time = 0

        for i, payload_file in enumerate(payload_list):
            payload = open(payload_file).read()

            logging.info(f"{i}/{total_payload_count} trying to upload: {payload_file}")

            uploaded, already_uploaded, upload_time = self.upload_payload(payload)

            uploaded_count += uploaded
            already_uploaded_count += already_uploaded
            total_upload_time += upload_time

        if already_uploaded_count > 0:
            logging.warning(f"{already_uploaded_count} payloads already uploaded.")
        logging.info(f"{uploaded_count}/{total_payload_count} Payload Uploaded in {total_upload_time}")
