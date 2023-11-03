import os
import tensorflow as tf
from xml.etree import ElementTree

from utils import log, Ccodes


class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, root_dir, data_split, image_size, batch_size, label_map, num_anchors, num_classes):
        self.root_dir = root_dir
        self.data_split = data_split  # "train" or "test"
        self.image_dir = os.path.join(root_dir, data_split, "images")
        self.annotation_dir = os.path.join(root_dir, data_split, "annotations")
        self.image_files = os.listdir(self.image_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.label_map = label_map
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = os.path.join(self.image_dir, self.image_files[idx])
        image = tf.image.decode_jpeg(tf.io.read_file(image_file), channels=3)

        annotation_file = os.path.join(self.annotation_dir, os.path.splitext(self.image_files[idx])[0] + ".xml")
        bounding_boxes = self.parse_xml_annotation(annotation_file)

        # Calculate the scaling factors
        original_size = tf.shape(image)[:2]
        original_height, original_width = tf.cast(original_size[0], dtype=tf.float32), tf.cast(original_size[1], dtype=tf.float32)
        scale_width = self.image_size[1] / original_width
        scale_height = self.image_size[0] / original_height

        # Resize the image
        image = tf.image.resize(image, list(self.image_size)[:2])
        image = tf.image.per_image_standardization(image)

        image = tf.expand_dims(image, axis=0)  # Add a batch dimension

        # Adjust bounding box coordinates [0, 1]
        adjusted_bounding_boxes = []
        for box in bounding_boxes["bounding_boxes"]:
            xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            xmin = tf.cast(xmin, tf.float32) / original_width
            xmax = tf.cast(xmax, tf.float32) / original_width
            ymin = tf.cast(ymin, tf.float32) / original_height
            ymax = tf.cast(ymax, tf.float32) / original_height
            adjusted_bounding_boxes.append([xmin, ymin, xmax, ymax])

        # log(f"- Bounding boxes: {adjusted_bounding_boxes}", Ccodes.GRAY)

        target_boxes = tf.convert_to_tensor(adjusted_bounding_boxes, dtype=tf.float32)

        labels = [box["name"] for box in bounding_boxes["bounding_boxes"]]
        label_indices = [self.label_map[label] for label in labels]

        # Create a one-hot encoded label for each bounding box
        y_true_classes = tf.convert_to_tensor([tf.one_hot(index, depth=self.num_classes) for index in label_indices], dtype=tf.float32)

        # If the number of objects in an image is less than num_anchors, pad y_true with zeros
        if tf.shape(y_true_classes)[0] < self.num_anchors:
            padding = tf.zeros([self.num_anchors - tf.shape(y_true_classes)[0], 4 + self.num_classes], dtype=tf.float32)
            y_true_classes = tf.concat([y_true_classes, padding[:, 4:]], axis=0)
            target_boxes = tf.concat([target_boxes, padding[:, :4]], axis=0)

        # Combine boxes and labels into a single tensor for the loss function
        y_true = tf.concat([target_boxes, y_true_classes], axis=-1)

        return image, y_true

    def parse_xml_annotation(self, annotation_file):
        # log(f"Parsing {annotation_file}", Ccodes.YELLOW)
        tree = ElementTree.parse(annotation_file)
        root = tree.getroot()

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        objects = root.findall("object")
        bounding_boxes = []
        for obj in objects:
            name = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            bounding_boxes.append({"name": name, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

        return {"width": width, "height": height, "bounding_boxes": bounding_boxes}
