import tensorflow as tf
import numpy as np
from . import cyclegan_datasets_mocap as cyclegan_datasets
from . import model_mocap as model
import csv
import pandas as pd
import math


def _load_samples(csv_name, image_type):
    filename_queue = tf.train.string_input_producer(
        [csv_name])

    reader = tf.TextLineReader()
    _, csv_filename = reader.read(filename_queue)

    record_defaults = [tf.constant([], dtype=tf.string),
                       tf.constant([], dtype=tf.string)]

    filename_i, filename_j = tf.decode_csv(
        csv_filename, record_defaults=record_defaults)


    record_defaults_csv = [tf.constant([], dtype=tf.float32)]*model.IMG_HEIGHT*model.IMG_WIDTH
    if image_type == '.csv':

        # dataset_a = tf.contrib.data.CsvDataset(file_contents_i,record_defaults_csv)
        #iter = dataset_a.make_one_shot_iterator()
       # next = iter.get_next()
       #
       #
       #  dataset= tf.contrib.data.CsvDataset(file_contents_j, record_defaults_csv,header = False )
       #  dataset = dataset.map(lambda *x: tf.convert_to_tensor(x))
       #  # dataset_batch = dataset.batch(16)
       #  iterator = dataset.make_initializable_iterator()
       #  image_decoded_a = iterator.get_next()

        #
        filename_queue_a = tf.train.string_input_producer(
            [filename_i])
        reader_a = tf.TextLineReader()

        _, csv_filename_a = reader_a.read(filename_queue_a)

        image_decoded_a = list(tf.decode_csv(
            csv_filename_a, record_defaults=record_defaults_csv,field_delim=','))

        # image_decoded_a = tf.stack( csv_return )



        filename_queue_b = tf.train.string_input_producer(
            [filename_j])
        reader_b = tf.TextLineReader()

        _, csv_filename_b = reader_b.read(filename_queue_b)

        image_decoded_b = list(tf.decode_csv(
            csv_filename_b, record_defaults=record_defaults_csv, field_delim=','))

        # image_decoded_a = tf.stack( csv_return )

        image_decoded_b = tf.reshape(
        image_decoded_b,
            [model.IMG_HEIGHT, model.IMG_WIDTH ],
            name=None
        )
        image_decoded_b = tf.transpose (image_decoded_b)

        image_decoded_a = tf.reshape(
            image_decoded_a,
            [model.IMG_HEIGHT, model.IMG_WIDTH ],
            name=None
        )
        image_decoded_a = tf.transpose(image_decoded_a)

        image_decoded_a = tf.expand_dims(tf.expand_dims(image_decoded_a, 0), 3)
        image_decoded_b = tf.expand_dims(tf.expand_dims(image_decoded_b, 0), 3)



        image_decoded_a_mean,image_decoded_a_sd= tf.nn.moments(image_decoded_a, axes=[1], keep_dims=True)
        image_decoded_b_mean, image_decoded_b_sd = tf.nn.moments(image_decoded_b, axes=[1], keep_dims=True)

        image_decoded_a =tf.div(tf.subtract(image_decoded_a, image_decoded_a_mean), image_decoded_a_sd)
        image_decoded_b= tf.div(tf.subtract(image_decoded_b, image_decoded_b_mean), mage_decoded_b_sd)

        image_decoded_a = tf.where(tf.is_nan(image_decoded_a), tf.zeros_like(image_decoded_a), image_decoded_a)
        image_decoded_b = tf.where(tf.is_nan(image_decoded_b), tf.zeros_like(image_decoded_b), image_decoded_b)



        # dataset = tf.data.experimental.CsvDataset(filename_i,
        #                                           record_defaults_csv, header=False)
        # dataset = dataset.map(lambda *x: tf.convert_to_tensor(x))
        # # dataset_batch = dataset.batch(16)
        # # iterator = dataset.make_initializable_iterator()
        # # image_decoded_a = iterator.get_next()
        # iterator = dataset.make_initializable_iterator()
        # image_decoded_a = iterator.get_next()
        # print('called load samples')
        #
        #
        #
        #
        #
        # dataset_j = tf.data.experimental.CsvDataset(filename_j,
        #                                           record_defaults_csv, header=False)
        # dataset_j = dataset_j.map(lambda *x: tf.convert_to_tensor(x))
        # # dataset_batch = dataset.batch(16)
        # # iterator = dataset.make_initializable_iterator()
        # # image_decoded_a = iterator.get_next()
        # iterator_j = dataset_j.make_initializable_iterator()
        # image_decoded_b = iterator_j.get_next()
        # print('called load samples')
        #





        # filename_queue_b = tf.train.string_input_producer(
        #     [file_contents_j])
        #
        # reader_b = tf.TextLineReader()
        # _, csv_filename_b = reader_b.read(filename_queue_b)
        # print("AFTER READER_B.READ")
        # mocap_b = tf.decode_csv(
        #     csv_filename_b, record_defaults=record_defaults_csv)
        # print("AFTER DECODE_B_CSV")
        # print(mocap_a)
        # image_decoded_a = tf.stack(mocap_a, axis = 0)
        #
        # image_decoded_b = tf.stack(mocap_b, axis = 0)
        # print("AFTER STACK_B")
        # print(image_decoded_a)

        #
        #
        #
        #
        # image_decoded_A = tf.decode_csv(
        #         file_contents_i, record_defaults=record_defaults)
        # image_decoded_B = tf.decode_csv(
        #         file_contents_j, record_defaults=record_defaults)

    # if image_type == '.jpg':
    #     image_decoded_A = tf.image.decode_jpeg(
    #         file_contents_i, channels=model.IMG_CHANNELS)
    #     image_decoded_B = tf.image.decode_jpeg(
    #         file_contents_j, channels=model.IMG_CHANNELS)
    # elif image_type == '.png':
    #     image_decoded_A = tf.image.decode_png(
    #         file_contents_i, channels=model.IMG_CHANNELS, dtype=tf.uint8)
    #     image_decoded_B = tf.image.decode_png(
    #         file_contents_j, channels=model.IMG_CHANNELS, dtype=tf.uint8)


    return image_decoded_a, image_decoded_b , image_decoded_a_mean,image_decoded_a_sd,image_decoded_b_mean, image_decoded_b_sd,  filename_i, filename_j


def load_data(dataset_name):
    """

    :param dataset_name: The name of the dataset.
    :param image_size_before_crop: Resize to this size before random cropping.
    :param do_shuffle: Shuffle switch.
    :param do_flipping: Flip switch.
    :return:
    """
    if dataset_name not in cyclegan_datasets.DATASET_TO_SIZES:
        raise ValueError('split name %s was not recognized.'
                         % dataset_name)

    csv_name = cyclegan_datasets.PATH_TO_CSV[dataset_name]

    image_i, image_j,image_i_mean,image_i_sd, image_j_mean,image_j_sd , filename_i, filename_j= _load_samples(
        csv_name, cyclegan_datasets.DATASET_TO_IMAGETYPE[dataset_name])


    inputs = {
        'images_i': image_i,
        'images_j': image_j,
        'images_i_mean': image_i_mean,
        'images_j_mean': image_j_mean,
        'images_i_sd': image_i_sd,
        'images_j_sd': image_j_sd,
        'images_i_name': filename_i,
        'images_j_name': filename_j,
    }

    # Preprocessing:
    # inputs['image_i'] = tf.image.resize_images(
    #     inputs['image_i'], [image_size_before_crop, image_size_before_crop])
    # inputs['image_j'] = tf.image.resize_images(
    #     inputs['image_j'], [image_size_before_crop, image_size_before_crop])
    #
    # if do_flipping is True:
    #     inputs['image_i'] = tf.image.random_flip_left_right(inputs['image_i'])
    #     inputs['image_j'] = tf.image.random_flip_left_right(inputs['image_j'])
    #
    # inputs['image_i'] = tf.random_crop(
    #     inputs['image_i'], [model.IMG_HEIGHT, model.IMG_WIDTH, 3])
    # inputs['image_j'] = tf.random_crop(
    #     inputs['image_j'], [model.IMG_HEIGHT, model.IMG_WIDTH, 3])

    # inputs['image_i'] = tf.subtract(tf.div(inputs['image_i'], 127.5), 1)
    # inputs['image_j'] = tf.subtract(tf.div(inputs['image_j'], 127.5), 1)

    #n Batch



    #
    # if do_shuffle is True:
    #     inputs['images_i'], inputs['images_j'] = tf.train.shuffle_batch(
    #         [inputs['image_i'], inputs['image_j']], 16, 5000, 100)
    # else:
    #     inputs['images_i'], inputs['images_j'] = tf.train.batch(
    #         [inputs['image_i'], inputs['image_j']], 16)



    #
    # if do_shuffle is True:
    #     inputs['images_i']= image_i
    #     inputs['images_j'] = image_j
    # else:
    #     inputs['images_i'] = image_i
    #     inputs['images_j'] = image_j

    return inputs