import os
import tensorflow as tf
import xml

from utils import log, Ccodes


class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, root_dir, data_split, image_size, batch_size, label_map):
        self.root_dir = root_dir
        self.data_split = data_split  # "train" or "test"
        self.image_dir = os.path.join(root_dir, data_split, "images")
        self.annotation_dir = os.path.join(root_dir, data_split, "annotations")
        self.image_files = os.listdir(self.image_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.label_map = label_map

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
        for box in bounding_boxes:
            xmin, ymin, xmax, ymax = box["boxes"]
            xmin = tf.cast(xmin, tf.float32) / original_width
            xmax = tf.cast(xmax, tf.float32) / original_width
            ymin = tf.cast(ymin, tf.float32) / original_height
            ymax = tf.cast(ymax, tf.float32) / original_height
            adjusted_bounding_boxes.append({"labels": box["labels"], "boxes": [xmin, ymin, xmax, ymax]})

        print("image shape", image.shape)
        print("adjusted bboxes", adjusted_bounding_boxes)
        log(f"- Bounding boxes: {adjusted_bounding_boxes}", Ccodes.GRAY)

        target_boxes = tf.convert_to_tensor([bb["boxes"] for bb in adjusted_bounding_boxes], dtype=tf.float32)

        labels = [label for bb in adjusted_bounding_boxes for label in bb["labels"]]
        label_indices = tf.convert_to_tensor([self.label_map[label] for label in labels], dtype=tf.int64)

        # Combine boxes and labels into a single tensor for the loss function
        y_true = tf.concat([target_boxes, tf.one_hot(label_indices, depth=len(self.label_map))], axis=-1)

        return image, y_true

    def parse_xml_annotation(self, xml_file):
        log(f"Parsing {xml_file}")
        tree = xml.etree.ElementTree.parse(xml_file)
        root = tree.getroot()

        bounding_boxes = []

        for obj in root.findall("object"):
            label = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            bounding_boxes.append({
                "labels": [label],
                "boxes": [xmin, ymin, xmax, ymax]
            })

        return bounding_boxes
