import tensorflow as tf
import numpy as np
# from . import cyclegan_datasets_mocap as cyclegan_datasets
# from . import model_mocap as model
import csv
import pandas as pd

def _load_samples(csv_name, image_type):
    filename_queue = tf.train.string_input_producer(
        [csv_name])

    reader = tf.TextLineReader()
    _, csv_filename = reader.read(filename_queue)

    record_defaults = [tf.constant([], dtype=tf.string),
                       tf.constant([], dtype=tf.string)]

    filename_i, filename_j = tf.decode_csv(
        csv_filename, record_defaults=record_defaults)
    filename = tf.decode_csv(
        csv_filename, record_defaults=record_defaults)


    file_contents_i = tf.read_file(filename_i)
    file_contents_j = tf.read_file(filename_j)


    record_defaults_csv = [tf.constant([], dtype=tf.float32)]*103
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
            [file_contents_i])

        reader_a = tf.TextLineReader()

        _, csv_filename_a = reader_a.read(filename_queue_a)
        print(filename_queue_a)

        csv_return = list(tf.decode_csv(
            csv_filename_a, record_defaults=record_defaults_csv,field_delim=','))

        image_decoded_a = tf.stack( csv_return )

        print (image_decoded_a)



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
    #
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

    return image_decoded_a

def main():

     record_defaults_csv = [tf.constant([], dtype=tf.float32)] * 103

     # dataset= tf.contrib.data.CsvDataset(file_contents_j, record_defaults_csv,header = False )
     dataset = tf.data.experimental.CsvDataset("./input/adult2child/train_adult/old_fast walking165.csv", record_defaults_csv, header=False)
     dataset = dataset.map(lambda *x: tf.convert_to_tensor(x))
     # dataset_batch = dataset.batch(16)
     # iterator = dataset.make_initializable_iterator()
     # image_decoded_a = iterator.get_next()
     iterator = dataset.make_initializable_iterator()
     next_element = iterator.get_next()
     print  (next_element)

     init = (tf.global_variables_initializer(),
             tf.local_variables_initializer())

     with tf.Session() as sess:
         sess.run(init)
         # Start populating the filename queue.
         coord = tf.train.Coordinator()
         threads = tf.train.start_queue_runners(coord=coord)
         sess.run(iterator.initializer)
         for i in range (0,4):
             # try:
                 print(sess.run(next_element))
             # except tf.errors.OutOfRangeError:
             #     break
         coord.request_stop()
         coord.join(threads)


# filename_queue = tf.train.string_input_producer(["./input/adult2child/train_adult/old_fast walking165.csv"])
    #
    # reader = tf.TextLineReader()
    # _, value = reader.read(filename_queue)
    #
    # # Default values, in case of empty columns. Also specifies the type of the
    # # decoded result.
    # record_defaults  = [tf.constant([], dtype=tf.float32)]*103
    # col1= list(tf.decode_csv(
    #     value, record_defaults=record_defaults))
    # features = tf.stack(col1)
    # col5 = tf.train.batch(features, 1)







        # Retrieve a single instance:


        # label = sess.run(col5)
        # print (label)


    # #image_decoded_a = _load_samples('./CycleGAN_TensorFlow/input/adult2child/train_adult/old_fast walking165.csv', '.csv')
    #
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #         sess.run(init)
    #
    #
    #         coord = tf.train.Coordinator()
    #         threads = tf.train.start_queue_runners(coord=coord)
    #
    #
    #         image_decoded_a = _load_samples(
    #             './CycleGAN_TensorFlow/input/adult2child/adult2child_train.csv', '.csv')
    #
    #
    #         inputs = tf.train.batch(image_decoded_a, 1)
    #         sess.run(inputs)
    #
    #
    #
    #          # Training Loop
    #         # for epoch in range(sess.run(self.global_step), self._max_step):
    #
    #
    #         coord.request_stop()
    #         coord.join(threads)

if __name__ == '__main__':
    main()
