import glob
import json
import logging
import os


def merge_payloads(first_payload_folder: str, second_payload_folder: str):
    if first_payload_folder.startswith('"'):
        first_payload_folder = first_payload_folder.strip('"')
    if second_payload_folder.startswith('"'):
        second_payload_folder = second_payload_folder.strip('"')

    if not os.path.isdir(first_payload_folder):
        exception = f"Path not found: {first_payload_folder}"
        logging.error(exception)
        raise Exception(exception)

    if not os.path.isdir(second_payload_folder):
        exception = f"Path not found: {second_payload_folder}"
        logging.error(exception)
        raise Exception(exception)

    first_payload_list = glob.glob(first_payload_folder + "/*.json")
    first_payload_list_count = len(first_payload_list)

    second_payload_list = glob.glob(second_payload_folder + "/*.json")
    second_payload_list_count = len(second_payload_list)

    if first_payload_list_count != second_payload_list_count:
        exception = f"Payload counts not equal!\n" \
                    f"{first_payload_folder}: {first_payload_list_count} payload\n" \
                    f"{second_payload_folder}: {second_payload_list_count} payload"
        logging.error(exception)
        raise Exception(exception)

    else:
        logging.info(f"Found payloads:\n"
                     f"{first_payload_folder}: {first_payload_list_count} payload\n"
                     f"{second_payload_folder}: {second_payload_list_count} payload")

    new_payload_folder = os.path.basename(first_payload_folder) + "__" + \
                         os.path.basename(second_payload_folder) + "_merged"

    new_payload_path = os.path.split(first_payload_folder)[0] + "/" + new_payload_folder + "/"

    if not os.path.isdir(new_payload_path):
        os.makedirs(new_payload_path)

    for first_payload, second_payload in zip(first_payload_list, second_payload_list):
        first_payload_json = json.load(open(first_payload))
        second_payload_json = json.load(open(second_payload))

        new_payload = {}

        if first_payload_json["frame"] != second_payload_json["frame"]:
            exception = f"Payload frame ids are not equal!\n" \
                        f"First payload: {first_payload}\n" \
                        f"Second payload: {second_payload}\n"

            logging.error(exception)
            raise Exception(exception)

        elif os.path.basename(first_payload) != os.path.basename(second_payload):
            exception = f"Payload file names are not equal!\n" \
                        f"First payload: {first_payload}\n" \
                        f"Second payload: {second_payload}\n"

            logging.error(exception)
            raise Exception(exception)

        else:
            new_payload["frame"] = first_payload_json["frame"]
            new_payload["detected_objects"] = first_payload_json["detected_objects"] + \
                                              second_payload_json["detected_objects"]

            json.dump(new_payload, open(new_payload_path + os.path.basename(first_payload), "w"))

    logging.info(f"Payloads merged in this folder: {new_payload_path}")
