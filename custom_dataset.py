import tensorflow as tf
from xml.etree import ElementTree
import os
import numpy as np
from utils import log, Ccodes

class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, dataset_dir, split, input_shape, batch_size, label_map, num_predictions=9):
        self.dataset_dir = dataset_dir
        self.split = split  # "train" or "test"
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.label_map = label_map
        self.image_dir = os.path.join(self.dataset_dir, self.split, "images")
        self.image_files = os.listdir(self.image_dir)
        self.num_classes = len(label_map)
        self.num_predictions = num_predictions
        self.predicted_boxes = tf.random.uniform([self.num_predictions, 4])

    def __len__(self):
        return int(np.ceil(len(self.image_files) / float(self.batch_size)))

    # def __len__(self):
    #     return (len(self.image_files) + self.batch_size - 1) // self.batch_size

    def iou(self, box1, box2):
        # Calculate the (x, y)-coordinates of the intersection rectangle
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        
        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)
        
        # Compute the area of both the prediction and ground-truth rectangles
        box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
        iou = interArea / float(box1Area + box2Area - interArea)
        return iou

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []

        for i in range(self.batch_size):
            image_file = os.path.join(self.image_dir, self.image_files[idx * self.batch_size + i])
            image = tf.image.decode_jpeg(tf.io.read_file(image_file), channels=3)
            
            annotation_file = os.path.join(
                self.dataset_dir, self.split, 
                "annotations", os.path.splitext(self.image_files[idx * self.batch_size + i])[0] + ".xml"
            )
            
            bounding_boxes = self.parse_xml_annotation(annotation_file, image)
            
            # Resize and standardize the image
            image = tf.image.resize(image, (self.input_shape[0], self.input_shape[1]))
            image = tf.image.per_image_standardization(image)
            batch_x.append(image)

            target_boxes = tf.convert_to_tensor([list(box.values()) for box in bounding_boxes["bounding_boxes"]], dtype=tf.float32)
            labels = bounding_boxes["labels"]
            label_indices = [self.label_map[label] for label in labels]
            
            # Let y_true be a tensor of zeros
            y_true = tf.zeros([49, self.num_predictions, self.num_classes + 5])
            target_boxes = target_boxes / [self.input_shape[1], self.input_shape[0], self.input_shape[1], self.input_shape[0]]

            # For each bounding box
            for box, label in zip(target_boxes, label_indices):
                # Determine which cell the bounding box belongs to
                cell_x = int(box[0] * 7)  # requires box coordinates be normalized
                cell_y = int(box[1] * 7)  # requires box coordinates be normalized
                cell_index = cell_y * 7 + cell_x

                ious = tf.convert_to_tensor([self.iou(box, predicted_box) for predicted_box in self.predicted_boxes])

                # Determine which prediction slot to use within the cell
                prediction_index = tf.argmax(ious)

                # Update the corresponding slot in y_true
                y_true = self.update_y_true(y_true, box, label, cell_index, prediction_index)

            batch_y.append(y_true)

        return np.array(batch_x), np.array(batch_y)

    def parse_xml_annotation(self, annotation_file, image):
        log(f"Processing {annotation_file}", Ccodes.YELLOW)
        tree = ElementTree.parse(annotation_file)
        root = tree.getroot()

        bounding_boxes = []
        labels = []

        # Get the original image dimensions
        original_width, original_height, _ = image.shape

        for obj in root.findall("object"):
            name = obj.find("name").text
            labels.append(name)

            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            # Normalize the bounding box coordinates
            xmin = int(bndbox.find("xmin").text) / original_width
            ymin = int(bndbox.find("ymin").text) / original_height
            xmax = int(bndbox.find("xmax").text) / original_width
            ymax = int(bndbox.find("ymax").text) / original_height

            bounding_boxes.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

        log(f"Bounding boxes: {bounding_boxes}", Ccodes.GREEN)
        return {"bounding_boxes": bounding_boxes, "labels": labels}
