from __future__ import print_function
import keras
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, DepthwiseConv2D, SeparableConv2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Input, add, GlobalMaxPooling2D, AveragePooling2D
from keras import optimizers, regularizers
import numpy as np
from keras.utils import plot_model
from keras import backend as K
import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
import time


class cifar10resnet18:

    def __init__(self, train=True):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32, 32, 3]
        self.epoches = 150
        self.batch_size = 100
        self.callback_list = self.get_callback_list()

        self.model = self.build_model()
        if train:

            start = time.time()
            self.train(self.model)
            end = time.time()

            print(f"共消耗了:{end - start}")
        else:
            self.model.load_weights('cifar10vgg.h5')

    @staticmethod
    def get_callback_list():

        callback_list = [
            ModelCheckpoint(filepath='cifar10_resnet18.h5',  # 目标文件的保存路径
                            monitor='val_accuracy',  # 监控验证损失
                            save_best_only=True),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5),
            TensorBoard('./logs/no_asym')
            # WandbCallback()
        ]

        return callback_list

    def conv_1(self, filters, input_layer):

        con = Conv2D(filters, kernel_size=3, padding='same')(input_layer)
        bn = BatchNormalization()(con)
        ac = Activation('relu')(bn)

        return ac

    def conv_2(self, pre_layer, filters):

        conv = Conv2D(filters=filters, kernel_size=3, padding='same')(pre_layer)
        bn = BatchNormalization()(conv)
        ac = Activation('relu')(bn)

        conv = Conv2D(filters=filters, kernel_size=3, padding='same')(ac)
        bn = BatchNormalization()(conv)
        ac1 = Activation('relu')(bn)

        add_ = add([ac, ac1])
        ac = Activation('relu')(add_)

        return ac

    def conv_3(self, pre_layer, filters, strides=1):

        if strides == 2:
            conv = Conv2D(filters, kernel_size=(3, 3), strides=2, padding='same')(pre_layer)
        else:
            conv = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(pre_layer)

        bn = BatchNormalization()(conv)
        ac = Activation('relu')(bn)
        conv = Conv2D(filters, kernel_size=3, padding='same')(ac)
        bn1 = BatchNormalization()(conv)

        if strides == 2:
            conv = Conv2D(filters, kernel_size=3, strides=2, padding='same')(pre_layer)
            bn2 = BatchNormalization()(conv)
            add_ = add([bn1, bn2])
        else:
            add_ = add([bn1, pre_layer])

        ac = Activation('relu')(add_)

        return ac

    def channel_shuffle(self, x, groups=1):

        height, width, in_channels = x.shape.as_list()[1:]
        channels_per_group = in_channels // groups

        x = K.reshape(x, [-1, height, width, groups, channels_per_group])
        x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
        x = K.reshape(x, [-1, height, width, in_channels])

        print(K.print_tensor(x, 'shuffled'))

        return x

    def gp_3x3(self, inputs, pre_layer):

        interval = int(pre_layer / 2)

        normalized1_1 = Lambda(lambda x: x[:, :, :, :interval])(inputs)
        normalized1_2 = Lambda(lambda x: x[:, :, :, interval:])(inputs)

        # 第一路
        tower_1 = Conv2D(interval,(3, 3), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_1)

        # 第二路
        tower_2 = Conv2D(interval,(3, 3), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_2)

        output = keras.layers.concatenate([tower_1, tower_2], axis=3)
        output = self.channel_shuffle(output, 2)

        output = Conv2D(pre_layer/4, 1, kernel_regularizer=regularizers.l2(self.weight_decay), activation='relu')(output)
        output = BatchNormalization()(output)

        return output

    def gp_5x5(self, inputs, pre_layer):
        interval = int(pre_layer / 4)

        normalized1_1 = Lambda(lambda x: x[:, :, :, :interval])(inputs)
        normalized1_2 = Lambda(lambda x: x[:, :, :, interval:2 * interval])(inputs)
        normalized1_3 = Lambda(lambda x: x[:, :, :, 2 * interval:3 * interval])(inputs)
        normalized1_4 = Lambda(lambda x: x[:, :, :, 3 * interval:])(inputs)

        # 第一路
        tower_1 = Conv2D(interval, (5, 5), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_1)

        # 第二路
        tower_2 = Conv2D(interval,(5, 5), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_2)

        # 第三路
        tower_3 = Conv2D(interval,(5, 5), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_3)

        # 第四路
        tower_4 = Conv2D(interval,(5, 5), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_4)
        output = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3)
        output = self.channel_shuffle(output, 4)

        output = Conv2D(pre_layer/4, 1, kernel_regularizer=regularizers.l2(self.weight_decay), activation='relu')(output)
        output = BatchNormalization()(output)

        return output

    def gp_7x7(self, inputs, pre_layer):
        interval = int(pre_layer / 4)

        normalized1_1 = Lambda(lambda x: x[:, :, :, :interval])(inputs)
        normalized1_2 = Lambda(lambda x: x[:, :, :, interval:2 * interval])(inputs)
        normalized1_3 = Lambda(lambda x: x[:, :, :, 2 * interval:3 * interval])(inputs)
        normalized1_4 = Lambda(lambda x: x[:, :, :, 3 * interval:])(inputs)

        # 第一路
        tower_1 = Conv2D(interval / 4, (7, 7), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_1)

        # 第二路
        tower_2 = Conv2D(interval / 4, (7, 7), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_2)

        # 第三路
        tower_3 = Conv2D(interval / 4, (7, 7), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_3)
        # 第四路
        tower_4 = Conv2D(interval / 4, (7, 7), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_4)

        output = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3)
        output = BatchNormalization()(output)

        return output

    def gp_block(self, inputs, in_filters):
        # 生成分组卷积模块

        # 第一路
        tower_1 = Conv2D(in_filters / 4, (1, 1), padding='same',
                         input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                         activation='relu')(inputs)
        tower_1 = BatchNormalization()(tower_1)

        # 第二路
        tower_2 = self.gp_3x3(inputs, in_filters)

        # 第三路
        tower_3 = self.gp_5x5(inputs, in_filters)

        # 第四路
        tower_4 = self.gp_7x7(inputs, in_filters)

        output = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3)

        return output

    def py_block(self, inputs, in_filters, out_filters):
        # 生成金字塔卷积模块

        weight_decay = 0.0005

        x = Conv2D(in_filters, 1, padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(
            inputs)
        x = BatchNormalization()(x)

        x = self.gp_block(x, in_filters)

        x = Conv2D(out_filters, 1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)

        inputs = Conv2D(out_filters, 1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        inputs = BatchNormalization()(inputs)

        x = keras.layers.add([inputs, x])
        x = Activation('relu')(x)

        return x

    def build_model(self):

        input_layer = Input(self.x_shape)

        con1 = self.conv_1(64, input_layer)

        con2 = self.conv_2(con1, 64)
        con2 = self.conv_2(con2, 64)

        con3 = self.conv_3(con2, 128, 2)
        con3 = self.py_block(con3, 128, 128)

        con4 = self.conv_3(con3, 256, 2)
        # conv4 = Dropout(.5)(con4)
        con4 = self.py_block(con4, 256, 256)
        con5 = self.conv_3(con4, 512, 2)
        # conv5 = Dropout(.5)(con5)
        con5 = self.py_block(con5, 512, 512)

        avp = AveragePooling2D(pool_size=4)(con5)
        flatten = Flatten()(avp)
        #
        # flatten = Flatten()(con5)
        # dense = Dense(256, activation='relu')(flatten)
        # dense = GlobalMaxPooling2D()(conv4)
        # dense = Dense(256, activation='relu')(dense)
        out_layer = Dense(self.num_classes, activation='softmax')(flatten)

        model = Model(input_layer, out_layer)

        plot_model(model, 'resnet18_2.png', show_shapes=True)


        model.summary()

        return model

    def normalize(self, X_train, X_test):
        # this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)
        return X_train, X_test

    def normalize_production(self, x):
        # this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        # these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x - mean) / (std + 1e-7)

    def predict(self, x, y, normalize=True, batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.evaluate(x, y, batch_size)

    def train(self, model):

        # training parameters

        learning_rate = 0.1
        lr_decay = 4e-5
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        # data augmentation
        datagen = ImageDataGenerator(
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        model.fit(datagen.flow(x_train, y_train,
                               batch_size=self.batch_size),
                  steps_per_epoch=x_train.shape[0] // self.batch_size,
                  epochs=self.epoches,
                  validation_data=(x_test, y_test),
                  validation_batch_size=128,
                  callbacks=self.callback_list)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = cifar10resnet18()

    loss, acc = model.predict(x_test, y_test, batch_size=128)

    print(f"验证集：准确率:{acc}, 损失:{loss}")





