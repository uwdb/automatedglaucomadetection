# input data pipeline definition
import tensorflow as tf
from random import random
import numpy as np

scales = list(np.arange(0.8, 1.0, 0.01))
boxes = np.zeros((len(scales), 4))

for i, scale in enumerate(scales):
    x1 = y1 = 0.5 - (0.5 * scale)
    x2 = y2 = 0.5 + (0.5 * scale)
    boxes[i] = [x1, y1, x2, y2]


def random_crop(img):
    # Create different crops for an image
    crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(210, 210))
    # Return a random crop
    return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


def _parse_train(filename, label, ls, nclasses=2, size=224):
    m = 27.5
    s = 32.3
    label_smoothing = 0.1
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image_decoded = tf.image.decode_png(image_string, channels=1)
    image = tf.cast(image_decoded, tf.float32)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)

    choice = random()
    if choice < 0.5:
        image = random_crop(image)

    image = tf.image.resize_images(image, [size, size])
    image = (image-m)/s

    ll = tf.one_hot(label, depth=nclasses)
    if ls is True:
        ll -= label_smoothing * (ll - 1. / nclasses)

    return image, ll


def _parse_test(filename, label,  nclasses=2):
    m = 27.5
    s = 32.3
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image_decoded = tf.image.decode_png(image_string, channels=1)
    image_decoded = tf.cast(image_decoded, tf.float32)

    # convert

    img = (image_decoded - m) / s
    ll = tf.one_hot(label, depth=nclasses)
    return img, ll


def input_fn(is_training, filenames, labels, batch_size, ls, nclasses, size):
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    # Create a Dataset serving batches of images and labels
    parse_train = lambda f, l,: _parse_train(f, l, ls, nclasses,size)
    parse_test = lambda f, l: _parse_test(f, l,  nclasses)

    if is_training:
        dataset = (
            tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                .shuffle(num_samples)
                .map(parse_train, num_parallel_calls=4)
                .batch(batch_size)
                .repeat()
                .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                   .map(parse_test)
                   .batch(batch_size)
                   .repeat()
                   .prefetch(1)  # make sure you always have one batch ready to serve
                   )

    return dataset
