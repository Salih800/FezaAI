class MODELS:
    def __init__(self):
        self.yolov7_e6e_yaya_arac = {"name": "yolov7-e6e-vismix+teknofest",
                                     "path": "models/yolov7-e6e-b8-e300-i960-vismix+teknofest.pt",
                                     "size": 960,
                                     "conf": 0.5,
                                     "model_for": "yaya-arac",
                                     "gdrive_id": "1-lbNSydY98tfvXRGAI8GAoZozX4wttnb",
                                     "which_yolo": "yolov7"
                                     }
        self.yolov7_uap_uai = {"name": "yolov7-teknofest+gta5",
                               "path": "models/yolov7-b8-e300-i1280-teknofest+gta5.pt",
                               "size": 1280,
                               "conf": 0.80,
                               "model_for": "uap-uai",
                               "gdrive_id": "1056zTMCVlDHqBIQztfa7biA5mzNIBH_i",
                               "which_yolo": "yolov7"
                               }
        self.yolov5s6_yaya_arac = {"name": "yolov5s6-vismix+teknofest",
                                   "path": "models/yolov5s6-b8-e300-i1920-vismix+teknofest.pt",
                                   "size": 1920,
                                   "conf": 0.5,
                                   "model_for": "yaya-arac",
                                   "gdrive_id": "10UEd74XaRrpf5L_8dPNlwV-tRRaDY4x1",
                                   "which_yolo": "yolov5"
                                   }
        self.yolov5s6_uap_uai = {"name": "yolov5s6-teknofest+gta5_uap-uai_v2",
                                 "path": "models/yolov5s6-b32-e300-i1280-teknofest+gta5_uap-uai_v2.pt",
                                 "size": 1280,
                                 "conf": 0.80,
                                 "gdrive_id": "10E2vIRVhB6o4IiKV9fZQcQxpDjx5Dafj",
                                 "model_for": "uap-uai",
                                 "which_yolo": "yolov5"
                                 }
        self.yolov5x6_yaya_arac = {"name": "yolov5x6-vismix+teknofest",
                                   "path": "models/yolov5x6-b8-e300-i1152-vismix+teknofest.pt",
                                   "size": 1152,
                                   "conf": 0.5,
                                   "gdrive_id": "1-nrbVsCKODyhIar-QFU4yksnvWurvdNl",
                                   "model_for": "yaya-arac",
                                   "which_yolo": "yolov5"
                                   }
        self.yolov5s_yaya_arac = {"name": "yolov5s-visdrone_model_v2",
                                  "path": "models/yolov5s-i800-visdrone_model_v2.pt",
                                  "size": 800,
                                  "conf": 0.5,
                                  "model_for": "yaya-arac",
                                  "gdrive_id": "10G3HTPJhV40hymlDieLJjvwjJO1G1Ix9",
                                  "which_yolo": "yolov5"
                                  }
        self.yolov5s_uap_uai = {"name": "yolov5s-uap-uai-v2",
                                "path": "models/yolov5s-i640-uap-uai-v2.pt",
                                "size": 640,
                                "conf": 0.8,
                                "gdrive_id": "10BoouFgKndIfnnMyNPR7nPjzSjBJM1ZI",
                                "model_for": "uap-uai",
                                "which_yolo": "yolov5"
                                }

        # self.yolov7_e6e = {"name": "yolov7-e6e-default",
        #                    "path": "models/yolov7-e6e-b8-e300-i960-vismix+teknofest.pt",
        #                    "size": 1280,
        #                    "conf": 0.5,
        #                    "model_for": "coco"
        #                    }
        # self.yolov7 = {"name": "yolov7-default",
        #                "path": "models/yolov7.pt",
        #                "size": 640,
        #                "conf": 0.5,
        #                "model_for": "coco"
        #                }
        # self.yolov5x6 = {"name": "yolov5x6-default",
        #                  "path": "models/yolov5x6.pt",
        #                  "size": 1280,
        #                  "conf": 0.5,
        #                  "model_for": "coco"
        #                  }
        # self.yolov5s6 = {"name": "yolov5s6-default",
        #                  "path": "models/yolov5s6.pt",
        #                  "size": 1280,
        #                  "conf": 0.5,
        #                  "model_for": "coco"
        #                  }
        # self.yolov5l6 = {"name": "yolov5l6-default",
        #                  "path": "models/yolov5l6.pt",
        #                  "size": 1280,
        #                  "conf": 0.5,
        #                  "model_for": "coco"
        #                  }


class ModelInfo:
    def __init__(self, model_info):
        self.name = model_info["name"]
        self.path = model_info["path"]
        self.image_size = model_info["size"]
        self.confidence_threshold = model_info["conf"]
        self.model_for = model_info["model_for"]
        self.gdrive_id = model_info["gdrive_id"]
        self.which_yolo = model_info["which_yolo"]


def get_model_info(model):
    return ModelInfo(model)
