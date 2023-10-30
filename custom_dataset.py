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
        scale_width = self.image_size[1] / tf.cast(original_size[1], dtype=tf.float32)
        scale_height = self.image_size[0] / tf.cast(original_size[0], dtype=tf.float32)

        # Resize the image
        image = tf.image.resize(image, self.image_size)
        image = tf.image.per_image_standardization(image)

        # Adjust bounding box coordinates
        adjusted_bounding_boxes = []
        for box in bounding_boxes:
            xmin, ymin, xmax, ymax = box["boxes"]
            xmin *= scale_width
            xmax *= scale_width
            ymin *= scale_height
            ymax *= scale_height
            adjusted_bounding_boxes.append({"labels": box["labels"], "boxes": [xmin, ymin, xmax, ymax]})

        log(f"- Bounding boxes: {adjusted_bounding_boxes}", Ccodes.GRAY)

        target_boxes = tf.convert_to_tensor([bb["boxes"] for bb in adjusted_bounding_boxes], dtype=tf.float32)

        labels = [label for bb in adjusted_bounding_boxes for label in bb["labels"]]
        label_indices = tf.convert_to_tensor([self.label_map[label] for label in labels], dtype=tf.int64)

        print(tf.shape(image), tf.shape(target_boxes), tf.shape(label_indices))
        return image, {"boxes": target_boxes, "labels": label_indices}

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
