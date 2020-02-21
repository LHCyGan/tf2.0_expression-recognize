# encoding:utf-8
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers, Sequential, metrics, optimizers, preprocessing
from resnet34 import ResNet34
from resnet50 import resnet50
import os


db_path = "./Training"
def gettraindb():
    image = []
    label = []
    n = 0
    # label_name = ["happy", "sad"]
    for i in os.listdir(db_path):
        path = os.path.join(db_path, i)
        for j in os.listdir(path):
            image.append(j)
            label.append(n)
        n += 1
    return image, label

# def preprocess(x, y):
#     x = tf.cast(x, tf.float32) / 255.
#     y = tf.cast(y, tf.int32)
#     return x, y

def preprocess_img(img):
    image = tf.image.decode_jpeg(img,)
    image = tf.image.resize(image, [224, 224])
    image /= 225.
    return image
def load_image(path):
    image = tf.io.read_file(path)
    return preprocess_img(image)

image_path, label_path = gettraindb()
# x, y = np.array(image), np.array(label)
# x = tf.convert_to_tensor(x)
# y = tf.convert_to_tensor(y)
def image_genetator():
    #建立生成器
    img_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1 / 255,
        validation_split=0.1,
        vertical_flip=True,
        rotation_range=45)
    #创建读取队列
    img_data = img_generator.flow_from_directory(os.path.join("./", 'Training'),
                                                 target_size=(48, 48),
                                                 batch_size=128)
    return img_data

if __name__ == '__main__':
    """Dataset"""
    # path_db = tf.data.Dataset.from_tensor_slices(tf.constant(image_path))
    # image_db = path_db.map(load_image)
    # label_db = tf.data.Dataset.from_tensor_slices(tf.cast(tf.constant(label_path), tf.int32))
    # train_db = tf.data.Dataset.zip((image_db, label_db))
    # train_db = train_db.batch(128).repeat(30).shuffle(998)
    #
    # sample = next(iter(train_db))
    #
    # model = ResNet34()
    # model.build(input_shape=(None, 48, 48, 1))
    # model.summary()
    #
    # model.compile(optimizer=optimizers.Adam(1e-2),
    #               loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    # model.fit(train_db, epochs=10, validation_split=0.1, validation_freq=2)

    """image_generator"""
    model = resnet50()
    # Inputs = keras.Input((48, 48, 3))
    # model(Inputs)
    model.build(input_shape=(None, 48, 48, 3))


    model.compile(optimizer=optimizers.Adam(1e-2),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # model.summary()
    # model.fit(image_genetator(), epochs=10, steps_per_epoch=10, validation_split=0.1, step_freq=2)
    model.fit(image_genetator(), epochs=10, steps_per_epoch=10)
    # model.save_weights("./expression_face.ckpt")