import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

class BasicBlock(layers.Layer):
    def __init__(self, filter_num, strides=1):
        super(BasicBlock, self).__init__()
        # ResBlock
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=strides, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        # Identify
        if strides > 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=strides, padding='same'))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        #resblock
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Identify

        Identify = self.downsample(inputs)

        #add
        output = layers.add([out, Identify])
        output = tf.nn.relu(output)

        return output

class ResNet(keras.Model):
    def __init__(self, layers_dim, num_classes=2):
        super(ResNet, self).__init__()
        #预处理
        self.stem = Sequential([
            layers.Conv2D(64, (3, 3), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same')
        ])
        #残差层
        self.layer_1 = self.Bulid_ResBlock(64, layers_dim[0])
        self.layer_2 = self.Bulid_ResBlock(128, layers_dim[1], strides=2)
        self.layer_3 = self.Bulid_ResBlock(256, layers_dim[2], strides=2)
        self.layer_4 = self.Bulid_ResBlock(512, layers_dim[3], strides=2)
        #全局平均池化和全连接层
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        out = self.stem(inputs)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)

        output = self.avgpool(out)
        output = self.fc(output)
        return output


    def Bulid_ResBlock(self, filter_num, blocks, strides=1): #建立残差层
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, strides))

        for i in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, strides))

        return res_blocks

def ResNet34():
    return ResNet([3, 4, 6, 3])