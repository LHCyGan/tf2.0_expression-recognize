#encoding=utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

#建立残差单元
class BasicBlock(layers.Layer):
    def __init__(self, filter_num1=0, filter_num2=0, strides=1, shortcut=False):
        super(BasicBlock, self).__init__()
        self.shortcut = shortcut

        self.conv1 = layers.Conv2D(filter_num1, (1, 1), strides=strides, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num1, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')

        self.conv3 = layers.Conv2D(filter_num2, (1, 1), strides=1, padding='same')
        self.bn3 = layers.Activation('relu')

        if strides >= 1:
            self.dowmsample = Sequential()
            self.dowmsample.add(layers.Conv2D(filter_num2, (1, 1), strides=1, padding='same'))
            self.dowmsample.add(layers.BatchNormalization())
        else:
            self.dowmsample = lambda x: x

    def call(self, inputs, training=None):
        out1 = self.conv1(inputs)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        out1 = self.conv2(inputs)
        out1 = self.bn2(out1)
        out1 = self.relu2(out1)
        out1 = self.conv3(inputs)
        out1 = self.bn3(out1)

        if self.shortcut == True:
            out2 = self.dowmsample(inputs)
        else:
            out2 = inputs

        output = layers.add([out1, out2])
        # output = tf.math.add(out1, out2)
        output = tf.nn.relu(output)
        return output

# # for 50, 101 or 152 layers
# class Block(keras.Model):
#
#     def __init__(self, filters, block_name,
#                  downsample=False, stride=1, **kwargs):
#         self.expasion = 4
#         super(Block, self).__init__(**kwargs)
#
#         conv_name = 'res' + block_name + '_branch'
#         bn_name = 'bn' + block_name + '_branch'
#         self.downsample = downsample
#
#         self.conv2a = keras.layers.Conv2D(filters=filters,
#                                           kernel_size=1,
#                                           strides=stride,
#                                           kernel_initializer='he_normal',
#                                           name=conv_name + '2a')
#         self.bn2a = keras.layers.BatchNormalization(axis=3, name=bn_name + '2a')
#
#         self.conv2b = keras.layers.Conv2D(filters=filters,
#                                           kernel_size=3,
#                                           padding='same',
#                                           kernel_initializer='he_normal',
#                                           name=conv_name + '2b')
#         self.bn2b = keras.layers.BatchNormalization(axis=3, name=bn_name + '2b')
#
#         self.conv2c = keras.layers.Conv2D(filters=4 * filters,
#                                           kernel_size=1,
#                                           kernel_initializer='he_normal',
#                                           name=conv_name + '2c')
#         self.bn2c = keras.layers.BatchNormalization(axis=3, name=bn_name + '2c')
#
#         if self.downsample:
#             self.conv_shortcut = keras.layers.Conv2D(filters=4 * filters,
#                                                      kernel_size=1,
#                                                      strides=stride,
#                                                      kernel_initializer='he_normal',
#                                                      name=conv_name + '1')
#             self.bn_shortcut = keras.layers.BatchNormalization(axis=3, name=bn_name + '1')
#
#     def call(self, inputs, **kwargs):
#         x = self.conv2a(inputs)
#         x = self.bn2a(x)
#         x = tf.nn.relu(x)
#
#         x = self.conv2b(x)
#         x = self.bn2b(x)
#         x = tf.nn.relu(x)
#
#         x = self.conv2c(x)
#         x = self.bn2c(x)
#
#         if self.downsample:
#             shortcut = self.conv_shortcut(inputs)
#             shortcut = self.bn_shortcut(shortcut)
#         else:
#             shortcut = inputs
#
#         x = keras.layers.add([x, shortcut])
#         x = tf.nn.relu(x)
#         return x


# 搭建网络
class ResNet(keras.Model):
    def __init__(self, layers_dim, num_classes=2):
        super(ResNet, self).__init__()

        self.padding = keras.layers.ZeroPadding2D((3, 3))
        self.stem = Sequential([
            layers.Conv2D(64, (7, 7), strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')
        ])

        self.layer1 = self.build_resblock(layers_dim[0], 64, 256, strides=1)
        self.layer2 = self.build_resblock(layers_dim[1], 128, 512, strides=2)
        self.layer3 = self.build_resblock(layers_dim[2], 256, 1024, strides=2)
        self.layer4 = self.build_resblock(layers_dim[3], 512, 2048, strides=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        out = self.padding(inputs)
        out = self.stem(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        output = self.fc(out)
        output = tf.nn.softmax(output)
        return output

    def build_resblock(self, blocks, filter_num1=0, filter_num2=0, strides=1):
        resblock = Sequential()
        if strides >= 1:
            resblock.add(BasicBlock(filter_num1, filter_num2, strides, shortcut=True))

        for _ in range(1, blocks):
            resblock.add(BasicBlock(filter_num1, filter_num2, strides=1))
        return resblock

# class ResNet(keras.Model):
#     def __init__(self, block, layers, num_classes=2, **kwargs):
#         super(ResNet, self).__init__(**kwargs)
#
#         self.padding = keras.layers.ZeroPadding2D((3, 3))
#         self.conv1 = keras.layers.Conv2D(filters=64,
#                                          kernel_size=7,
#                                          strides=2,
#                                          kernel_initializer='glorot_uniform',
#                                          name='conv1')
#         self.bn_conv1 = keras.layers.BatchNormalization(axis=3, name='bn_conv1')
#         self.max_pool = keras.layers.MaxPooling2D((3, 3),
#                                                   strides=2,
#                                                   padding='same')
#         self.avgpool = keras.layers.GlobalAveragePooling2D(name='avg_pool')
#         self.fc = keras.layers.Dense(num_classes, activation='softmax', name='result')
#
#         # layer2
#         self.res2 = self.mid_layer(block, 64, layers[0], stride=1, layer_number=2)
#
#         # layer3
#         self.res3 = self.mid_layer(block, 128, layers[1], stride=2, layer_number=3)
#
#         # layer4
#         self.res4 = self.mid_layer(block, 256, layers[2], stride=2, layer_number=4)
#
#         # layer5
#         self.res5 = self.mid_layer(block, 512, layers[3], stride=2, layer_number=5)
#
#     def mid_layer(self, block, filter, block_layers, stride=1, layer_number=1):
#         layer = keras.Sequential()
#         if stride != 1 or filter * 4 != 64:
#             layer.add(block(filters=filter,
#                             downsample=True, stride=stride,
#                             block_name='{}a'.format(layer_number)))
#
#         for i in range(1, block_layers):
#             p = chr(i + ord('a'))
#             layer.add(block(filters=filter,
#                             block_name='{}'.format(layer_number) + p))
#
#         return layer
#
#     def call(self, inputs, **kwargs):
#         x = self.padding(inputs)
#         x = self.conv1(x)
#         x = self.bn_conv1(x)
#         x = tf.nn.relu(x)
#         x = self.max_pool(x)
#
#         # layer2
#         x = self.res2(x)
#         # layer3
#         x = self.res3(x)
#         # layer4
#         x = self.res4(x)
#         # layer5
#         x = self.res5(x)
#
#         x = self.avgpool(x)
#         x = self.fc(x)
#         return x
# def resnet50():
#     return ResNet(Block, [3, 4, 6, 3], num_classes=2)

def resnet50():
    return ResNet([3, 4, 6, 3])