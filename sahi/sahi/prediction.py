# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import copy
import os.path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

from sahi.annotation import ObjectAnnotation
from sahi.utils.coco import CocoAnnotation, CocoPrediction
from sahi.utils.cv import read_image_as_pil, visualize_object_predictions
from sahi.utils.file import Path
from src.detected_object import DetectedObject

class PredictionScore:
    def __init__(self, value: float):
        """
        Arguments:
            score: prediction score between 0 and 1
        """
        # if score is a numpy object, convert it to python variable
        if type(value).__module__ == "numpy":
            value = copy.deepcopy(value).tolist()
        # set score
        self.value = value

    def is_greater_than_threshold(self, threshold):
        """
        Check if score is greater than threshold
        """
        return self.value > threshold

    def __repr__(self):
        return f"PredictionScore: <value: {self.value}>"


class ObjectPrediction(ObjectAnnotation):
    """
    Class for handling detection model predictions.
    """

    def __init__(
        self,
        bbox: Optional[List[int]] = None,
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        bool_mask: Optional[np.ndarray] = None,
        score: Optional[float] = 0,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Creates ObjectPrediction from bbox, score, category_id, category_name, bool_mask.

        Arguments:
            bbox: list
                [minx, miny, maxx, maxy]
            score: float
                Prediction score between 0 and 1
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            bool_mask: np.ndarray
                2D boolean mask array. Should be None if model doesn't output segmentation mask.
            shift_amount: list
                To shift the box and mask predictions from sliced image
                to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: list
                Size of the full image after shifting, should be in
                the form of [height, width]
        """
        self.score = PredictionScore(score)
        super().__init__(
            bbox=bbox,
            category_id=category_id,
            bool_mask=bool_mask,
            category_name=category_name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    def get_shifted_object_prediction(self):
        """
        Returns shifted version ObjectPrediction.
        Shifts bbox and mask coords.
        Used for mapping sliced predictions over full image.
        """
        if self.mask:
            return ObjectPrediction(
                bbox=self.bbox.get_shifted_box().to_voc_bbox(),
                category_id=self.category.id,
                score=self.score.value,
                bool_mask=self.mask.get_shifted_mask().bool_mask,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_shape=self.mask.get_shifted_mask().full_shape,
            )
        else:
            return ObjectPrediction(
                bbox=self.bbox.get_shifted_box().to_voc_bbox(),
                category_id=self.category.id,
                score=self.score.value,
                bool_mask=None,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_shape=None,
            )

    def to_coco_prediction(self, image_id=None):
        """
        Returns sahi.utils.coco.CocoPrediction representation of ObjectAnnotation.
        """
        if self.mask:
            coco_prediction = CocoPrediction.from_coco_segmentation(
                segmentation=self.mask.to_coco_segmentation(),
                category_id=self.category.id,
                category_name=self.category.name,
                score=self.score.value,
                image_id=image_id,
            )
        else:
            coco_prediction = CocoPrediction.from_coco_bbox(
                bbox=self.bbox.to_coco_bbox(),
                category_id=self.category.id,
                category_name=self.category.name,
                score=self.score.value,
                image_id=image_id,
            )
        return coco_prediction

    def to_fiftyone_detection(self, image_height: int, image_width: int):
        """
        Returns fiftyone.Detection representation of ObjectPrediction.
        """
        try:
            import fiftyone as fo
        except ImportError:
            raise ImportError('Please run "pip install -U fiftyone" to install fiftyone first for fiftyone conversion.')

        x1, y1, x2, y2 = self.bbox.to_voc_bbox()
        rel_box = [x1 / image_width, y1 / image_height, (x2 - x1) / image_width, (y2 - y1) / image_height]
        fiftyone_detection = fo.Detection(label=self.category.name, bounding_box=rel_box, confidence=self.score.value)
        return fiftyone_detection

    def __repr__(self):
        return f"""ObjectPrediction<
    bbox: {self.bbox},
    mask: {self.mask},
    score: {self.score},
    category: {self.category}>"""


class PredictionResult:
    def __init__(
        self,
        object_prediction_list: List[ObjectPrediction],
        image: Union[Image.Image, str, np.ndarray],
        durations_in_seconds: Optional[Dict] = None,
    ):
        self.image: Image.Image = read_image_as_pil(image)
        self.image_width, self.image_height = self.image.size
        self.object_prediction_list: List[ObjectPrediction] = object_prediction_list
        self.durations_in_seconds = durations_in_seconds

    def export_visuals(
        self, export_dir: str, text_size: float = None, rect_th: int = None, file_name: str = "prediction_visual"
    ):
        """

        Args:
            export_dir: directory for resulting visualization to be exported
            text_size: size of the category name over box
            rect_th: rectangle thickness
            file_name: saving name
        Returns:

        """
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        visualize_object_predictions(
            image=np.ascontiguousarray(self.image),
            object_prediction_list=self.object_prediction_list,
            rect_th=rect_th,
            text_size=text_size,
            text_th=None,
            color=None,
            output_dir=export_dir,
            file_name=file_name,
            export_format="png",
        )

    def to_coco_annotations(self):
        coco_annotation_list = []
        for object_prediction in self.object_prediction_list:
            coco_annotation_list.append(object_prediction.to_coco_prediction().json)
        return coco_annotation_list

    def to_coco_predictions(self, image_id: Optional[int] = None):
        coco_prediction_list = []
        for object_prediction in self.object_prediction_list:
            coco_prediction_list.append(object_prediction.to_coco_prediction(image_id=image_id).json)
        return coco_prediction_list

    def to_imantics_annotations(self):
        imantics_annotation_list = []
        for object_prediction in self.object_prediction_list:
            imantics_annotation_list.append(object_prediction.to_imantics_annotation())
        return imantics_annotation_list

    def to_fiftyone_detections(self):
        try:
            import fiftyone as fo
        except ImportError:
            raise ImportError('Please run "pip install -U fiftyone" to install fiftyone first for fiftyone conversion.')

        fiftyone_detection_list: List[fo.Detection] = []
        for object_prediction in self.object_prediction_list:
            fiftyone_detection_list.append(
                object_prediction.to_fiftyone_detection(image_height=self.image_height, image_width=self.image_width)
            )
        return fiftyone_detection_list

    def to_teknofest_predictions(self) -> list[DetectedObject]:
        detected_object_list: list[DetectedObject] = []
        for object_detection in self.object_prediction_list:
            cls = object_detection.category.id
            top_left_x = object_detection.bbox.minx
            top_left_y = object_detection.bbox.miny
            bottom_right_x = object_detection.bbox.maxx
            bottom_right_y = object_detection.bbox.maxy
            if object_detection.category.name in ["uap", "uai"]:
                landing_status = 1
            elif object_detection.category.name in ["uap_not", "uai_not"]:
                landing_status = 0
            else:
                landing_status = -1
            detected_object = DetectedObject(cls=cls, landing_status=landing_status, top_left_x=top_left_x, top_left_y=top_left_y,
                           bottom_right_x=bottom_right_x, bottom_right_y=bottom_right_y)
            detected_object_list.append(detected_object)
        return detected_object_list


    def save_yolo_label(self, label_path):
        label_folder = os.path.split(label_path)[0]
        if not os.path.isdir(label_folder):
           # shutil.rmtree(label_folder)
            os.makedirs(label_folder)

        for object_detection in self.object_prediction_list:
            cls = object_detection.category.id
            x = ((object_detection.bbox.minx + object_detection.bbox.maxx) / 2) / self.image_width
            y = ((object_detection.bbox.miny + object_detection.bbox.maxy) / 2) / self.image_height
            w = (object_detection.bbox.maxx - object_detection.bbox.minx) / self.image_width
            h = (object_detection.bbox.maxy - object_detection.bbox.miny) / self.image_height
            yolo_label = f"{cls} {x} {y} {w} {h}\n"
            with open(label_path, "a") as label_file:
                label_file.write(yolo_label)
            # break
