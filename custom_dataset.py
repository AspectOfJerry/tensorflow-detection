import tensorflow as tf
from xml.etree import ElementTree
import os
import numpy as np


class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, dataset_dir, split, input_shape, batch_size, label_map):
        self.dataset_dir = dataset_dir
        self.split = split  # "train" or "test"
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.label_map = label_map
        self.image_dir = os.path.join(self.dataset_dir, self.split, "images")
        self.image_files = os.listdir(self.image_dir)
        self.num_classes = len(label_map)
        self.num_anchors = 9  # max number of objects per image

    def __len__(self):
        return int(np.ceil(len(self.image_files) / float(self.batch_size)))

    # def __len__(self):
    #     return (len(self.image_files) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []

        for i in range(self.batch_size):
            image_file = os.path.join(self.image_dir, self.image_files[idx * self.batch_size + i])
            image = tf.image.decode_jpeg(tf.io.read_file(image_file), channels=3)

            annotation_file = os.path.join(self.dataset_dir, self.split, "annotations",
                                           os.path.splitext(self.image_files[idx * self.batch_size + i])[0] + ".xml")
            bounding_boxes = self.parse_xml_annotation(annotation_file, image)

            # Resize and standardize the image
            image = tf.image.resize(image, (self.input_shape[0], self.input_shape[1]))
            image = tf.image.per_image_standardization(image)

            target_boxes = tf.convert_to_tensor([list(box.values()) for box in bounding_boxes["bounding_boxes"]], dtype=tf.float32)
            labels = bounding_boxes["labels"]
            label_indices = [self.label_map[label] for label in labels]

            # Create a one-hot encoded label for each bounding box
            y_true_classes = tf.convert_to_tensor([tf.one_hot(index, depth=self.num_classes) for index in label_indices], dtype=tf.float32)

            # If the number of objects in an image is less than num_anchors, pad y_true with zeros
            if tf.shape(y_true_classes)[0] < self.num_anchors:
                padding = tf.zeros([self.num_anchors - tf.shape(y_true_classes)[0], self.num_classes], dtype=tf.float32)
                y_true_classes = tf.concat([y_true_classes, padding], axis=0)
                target_boxes = tf.concat([target_boxes, tf.zeros([self.num_anchors - tf.shape(target_boxes)[0], 4])], axis=0)

            # Combine boxes and labels into a single tensor for the loss function
            y_true = tf.concat([target_boxes, y_true_classes], axis=-1)

            batch_x.append(image)
            batch_y.append(y_true)

        return tf.stack(batch_x), tf.stack(batch_y)

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

            bounding_boxes.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

        return {"bounding_boxes": bounding_boxes, "labels": labels}
