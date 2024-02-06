import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub
import numpy as np
import PIL.Image


def load_image(infilename, max_dim):
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


def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


max_dim = 2048
content_image = load_image('C:\\Users\\Thomas\PycharmProjects\\MachineLearningPlayground2\\Portfolio\\ImageStyleTransfer\\Images\\base_image_test_2.png', max_dim=max_dim)
style_image = load_image('C:\\Users\\Thomas\PycharmProjects\\MachineLearningPlayground2\\Portfolio\\ImageStyleTransfer\\Images\\high_tech_image_1.jpg', max_dim=max_dim)

hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
tensor_to_image_stylized_image = tensor_to_image(stylized_image)
tensor_to_image_stylized_image.save('Test_Output_Image_19.jpg')