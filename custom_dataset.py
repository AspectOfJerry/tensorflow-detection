import os
import tensorflow as tf
from xml.etree import ElementTree

from utils import log, Ccodes


def load_and_preprocess_data(image_file, annotation_file, image_size, label_map, num_anchors, num_classes):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size[0], image_size[1]])
    image = image / 255.0  # normalize to [0,1] range

    bounding_boxes = tf.numpy_function(parse_xml_annotation, [annotation_file], tf.float32)
    bounding_boxes.set_shape([None, 4])

    # Calculate the scaling factors
    original_size = tf.shape(image)[:2]
    original_height, original_width = tf.cast(original_size[0], dtype=tf.float32), tf.cast(original_size[1], dtype=tf.float32)
    scale_width = image_size[1] / original_width
    scale_height = image_size[0] / original_height

    # Resize the image
    image = tf.image.resize(image, list(image_size)[:2])
    image = tf.image.per_image_standardization(image)

    # Adjust bounding box coordinates [0, 1]
    adjusted_bounding_boxes = []
    for box in bounding_boxes["bounding_boxes"]:
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        xmin = tf.cast(xmin, tf.float32) / original_width
        xmax = tf.cast(xmax, tf.float32) / original_width
        ymin = tf.cast(ymin, tf.float32) / original_height
        ymax = tf.cast(ymax, tf.float32) / original_height
        adjusted_bounding_boxes.append([xmin, ymin, xmax, ymax])

    target_boxes = tf.convert_to_tensor(adjusted_bounding_boxes, dtype=tf.float32)

    labels = [box["name"] for box in bounding_boxes["bounding_boxes"]]
    label_indices = [label_map[label] for label in labels]

    # Create a one-hot encoded label for each bounding box
    y_true_classes = tf.convert_to_tensor([tf.one_hot(index, depth=num_classes) for index in label_indices], dtype=tf.float32)

    # If the number of objects in an image is less than num_anchors, pad y_true with zeros
    if tf.shape(y_true_classes)[0] < num_anchors:
        padding = tf.zeros([num_anchors - tf.shape(y_true_classes)[0], 4 + num_classes], dtype=tf.float32)
        y_true_classes = tf.concat([y_true_classes, padding[:, 4:]], axis=0)
        target_boxes = tf.concat([target_boxes, padding[:, :4]], axis=0)

    # Combine boxes and labels into a single tensor for the loss function
    y_true = tf.concat([target_boxes, y_true_classes], axis=-1)

    return image, y_true


def parse_xml_annotation(annotation_file):
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

    # Convert bounding_boxes to a tensor
    bounding_boxes = [[box["xmin"], box["ymin"], box["xmax"], box["ymax"]] for box in bounding_boxes]
    bounding_boxes_tensor = tf.convert_to_tensor(bounding_boxes, dtype=tf.float32)

    return bounding_boxes_tensor


def create_dataset(root_dir, data_split, image_size, batch_size, label_map, num_anchors, num_classes):
    image_dir = os.path.join(root_dir, data_split, "images")
    annotation_dir = os.path.join(root_dir, data_split, "annotations")

    image_files = tf.data.Dataset.list_files(os.path.join(image_dir, '*.jpg'))
    annotation_files = tf.data.Dataset.list_files(os.path.join(annotation_dir, '*.xml'))

    dataset = tf.data.Dataset.zip((image_files, annotation_files))
    dataset = dataset.map(lambda x, y: load_and_preprocess_data(x, y, image_size, label_map, num_anchors, num_classes))
    dataset = dataset.batch(batch_size)

    return dataset
