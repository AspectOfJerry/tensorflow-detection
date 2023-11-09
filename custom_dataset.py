import tensorflow as tf
import os
from xml.etree import ElementTree
import numpy as np


class CustomDataset:
    def __init__(self, dataset_dir, split, input_shape, batch_size, label_map, anchors):
        self.dataset_dir = dataset_dir
        self.split = split  # "train" or "test"
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.label_map = label_map
        self.anchors = anchors

    def iou(self, box1, box2):
        # Calculate the (x, y)-coordinates of the intersection rectangle
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Compute the area of both the prediction and ground-truth rectangles
        box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # Compute the intersection over union by taking the intersection area and
        # dividing it by the sum of prediction + ground-truth areas - the intersection area
        iou = interArea / float(box1Area + box2Area - interArea)

        return iou

    def load_data(self):
        def generator():
            for image_file in os.listdir(os.path.join(self.dataset_dir, self.split, "images")):
                image_path = os.path.join(self.dataset_dir, self.split, "images", image_file)
                image = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)

                annotation_file = os.path.join(self.dataset_dir, self.split, "annotations", os.path.splitext(image_file)[0] + ".xml")
                bounding_boxes, labels = self.parse_xml_annotation(annotation_file, image)

                # Resize and preprocess the image
                image = tf.image.resize(image, self.input_shape[:2])
                image = tf.image.per_image_standardization(image)

                # Create one-hot encoded labels and bounding box coordinates for each bounding box
                y_true = np.zeros((len(self.anchors), len(self.label_map) + 4))  # +4 for bounding box coordinates
                for bbox, label in zip(bounding_boxes, labels):
                    label_index = self.label_map[label]
                    iou_scores = [self.iou(bbox, anchor) for anchor in self.anchors]
                    anchor_index = np.argmax(iou_scores)
                    y_true[anchor_index, label_index] = 1  # one-hot encoded class label
                    y_true[anchor_index, -4:] = bbox  # bounding box coordinates

                yield image, y_true

        return tf.data.Dataset.from_generator(generator, output_signature=(
            tf.TensorSpec(shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=tf.float32),
            tf.TensorSpec(shape=(len(self.anchors), len(self.label_map) + 4), dtype=tf.float32)))  # +4 for bounding box coordinates

    def parse_xml_annotation(self, annotation_file, image):
        tree = ElementTree.parse(annotation_file)
        root = tree.getroot()

        bounding_boxes = []
        labels = []

        original_height, original_width = image.shape[0], image.shape[1]

        for obj in root.findall("object"):
            name = obj.find("name").text
            labels.append(name)

            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            # Resize bounding box coordinates based on the new image dimensions
            xmin = xmin * self.input_shape[1] // original_width
            ymin = ymin * self.input_shape[0] // original_height
            xmax = xmax * self.input_shape[1] // original_width
            ymax = ymax * self.input_shape[0] // original_height

            bounding_boxes.append([xmin, ymin, xmax, ymax])

        return bounding_boxes, labels
