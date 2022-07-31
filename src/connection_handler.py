import datetime
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
            images_json_filename = datetime.datetime.now().strftime(images_json_path + self.frames[0]["video_name"]
                                                                    + '__%Y_%m_%d__%H_%M_%S_%f.json')
            with open(images_json_filename, "w") as json_file:
                json.dump(self.frames, json_file)
            logging.info("Successful : get_frames : {} pictures saved in the file: {}".format(len(self.frames),
                                                                                              images_json_filename))
        else:
            logging.info("Failed : get_frames : {}".format(response.text))

        return self.frames

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
        files = []
        headers = {
            'Authorization': 'Token {}'.format(self.auth_token),
            'Content-Type': 'application/json',
        }

        if upload_payload:
            response = requests.request("POST", self.url_prediction, headers=headers, data=payload, files=files)

            if response.status_code == 201:
                logging.info(f"Prediction uploaded successfully.")

            else:
                logging.warning("Prediction send failed. {}\n\t{}".format(response.status_code, response.text))
                response_json = json.loads(response.text)
                if "You do not have permission to perform this action." in response_json["detail"]:
                    logging.warning("Limit exceeded. 80frames/min \n\t{}".format(response.text))

        if save_payload:
            payloads_path = "./_payloads/" + prediction.video_name + "/" + model_name + "/"
            if not os.path.isdir(payloads_path):
                os.makedirs(payloads_path)

            payload_filename = payloads_path + os.path.basename(prediction.image_url)[:-4] + ".json"

            with open(payload_filename, "w") as detection_file:
                detection_file.write(payload)
            logging.info(f"Payload file saved: {payload_filename}")

        return response

    def upload_payloads(self, payload_folder: str):
        payload_list = glob.glob(payload_folder + "*.json")
        total_payload_count = len(payload_list)
        logging.info(f"{total_payload_count} Found Total Payloads in '{payload_folder}'")
        uploaded_count = 0
        failed_count = 0
        already_uploaded_count = 0

        for payload_file in payload_list:
            payload = open(payload_file).read()
            payload_uploaded = False
            while not payload_uploaded:
                files = []
                headers = {
                    'Authorization': 'Token {}'.format(self.auth_token),
                    'Content-Type': 'application/json',
                }
                response = requests.request("POST", self.url_prediction, headers=headers, data=payload, files=files)

                if response.status_code == 201:
                    logging.info(f"Prediction uploaded successfully.")
                    # payload_list.remove(payload_file)
                    uploaded_count += 1
                    payload_uploaded = True

                elif response.status_code == 406:
                    # payload_list.remove(payload_file)
                    already_uploaded_count += 1
                    payload_uploaded = True

                elif response.status_code == 403:
                    logging.warning("Limit exceeded. 80frames/min \n\t{}".format(response.text))
                    logging.info("Waiting 5 seconds...")
                    time.sleep(5)

                else:
                    failed_count += 1
                    logging.warning("Prediction send failed. {}\n\t{}".format(response.status_code, response.text))
                    response_json = json.loads(response.text)
                    if "You do not have permission to perform this action." in response_json["detail"]:
                        logging.warning("Limit exceeded. 80frames/min \n\t{}".format(response.text))
                    time.sleep(3)

        logging.info(f"{uploaded_count}/{total_payload_count} Payload Uploaded")
        logging.warning(f"{already_uploaded_count} payloads already uploaded and {failed_count} failed uploads!")
