import cv2
import tensorflow as tf
from keras.preprocessing.image import load_img
import numpy as np
import PIL

def load_image( infilename ) :
    img = PIL.Image.open(infilename)
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def load_image_2(infilename):
    max_dim = 1028
    img = tf.io.read_file(infilename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


im = load_image_2('output/Output_1.jpg')