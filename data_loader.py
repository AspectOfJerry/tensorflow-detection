import tensorflow as tf
from pycocotools.coco import COCO
import os

def load_coco_data(dataset_dir, category_names, batch_size):
    # Load COCO dataset
    coco = COCO(os.path.join(dataset_dir, "result.json"))

    # Get category IDs
    category_ids = coco.getCatIds(catNms=category_names)

    # Get image IDs for images that contain all categories
    image_ids = coco.getImgIds(catIds=category_ids)

    # Split into training and testing datasets
    train_image_ids = image_ids[:int(0.8 * len(image_ids))]
    test_image_ids = image_ids[int(0.8 * len(image_ids)):]

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(train_image_ids)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_image_ids)

    # Function to load and preprocess a single image
    def load_single_image(image_id):
        image_info = coco.loadImgs(image_id.numpy())[0]
        image = tf.io.read_file(os.path.join(dataset_dir, image_info['file_name']))
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [756, 756])
        image = (image - 127.5) / 127.5  # Normalize to [-1,1]

        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=image_info['id'], catIds=category_ids, iscrowd=None)
        annotations = coco.loadAnns(ann_ids)

        # Convert annotations to your desired format, and return image and annotations
        print(annotations)
        exit()
        return image, annotations

    # Use tf.py_function to allow numpy operations
    def load_single_image_wrapper(image_id):
        return tf.py_function(load_single_image, [image_id], [tf.float32])

    # Map loading function to train and test datasets
    train_dataset = train_dataset.map(load_single_image_wrapper)
    test_dataset = test_dataset.map(load_single_image_wrapper)

    # Batch datasets
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset
