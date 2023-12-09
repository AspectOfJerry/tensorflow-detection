import tensorflow as tf
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split


def load_coco_data(dataset_dir, category_names, batch_size, test_size=0.2, random_state=42):
    annotations_path = f"{dataset_dir}/result.json"
    image_dir = f"{dataset_dir}/images/"

    # Load the annotations file
    coco = COCO(annotations_path)
    category_ids = coco.getCatIds(catNms=category_names)
    print(category_ids)

    # Get the IDs of the images that contain either a 'cone' or a 'cube'
    image_ids = []
    for category_id in category_ids:
        image_ids.extend(coco.getImgIds(catIds=[category_id]))
    image_ids = list(set(image_ids))  # Remove duplicates
    print(image_ids)

    # Split the data into training and testing sets
    train_image_ids, test_image_ids = train_test_split(image_ids, test_size=test_size, random_state=random_state)

    def load_single_image(image_id):
        image_id = image_id.numpy()  # Convert the tensor to a Python type
        image_info = coco.loadImgs(image_id)[0]
        image_path = tf.strings.join([image_dir, image_info["file_name"]])
        ann_ids = coco.getAnnIds(imgIds=image_id, catIds=category_ids, iscrowd=None)
        annotations = coco.loadAnns(ann_ids)

        # Load and normalize the image
        image = tf.image.decode_image(tf.io.read_file(image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (756, 756))
        image = (image - 0.5) / 0.5  # Normalize to the range [-1, 1]

        return image, annotations

    def load_single_image_wrapper(image_id):
        image, annotations = tf.py_function(load_single_image, [image_id], [tf.float32, tf.float32])
        image.set_shape([756, 756, 3])  # Set the shape of the image tensor
        return image, annotations


    # Use train_image_ids for the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(train_image_ids)
    # train_dataset = train_dataset.map(load_single_image)
    train_dataset = train_dataset.map(load_single_image_wrapper)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_image_ids))
    train_dataset = train_dataset.batch(batch_size)

    # Use test_image_ids for the testing dataset
    test_dataset = tf.data.Dataset.from_tensor_slices(test_image_ids)
    test_dataset = test_dataset.map(load_single_image)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset
