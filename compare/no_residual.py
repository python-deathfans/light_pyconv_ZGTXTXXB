from __future__ import print_function
import keras
from keras.datasets import cifar100, cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, GlobalMaxPooling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, SeparableConv2D, DepthwiseConv2D, SpatialDropout2D
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import wandb
from wandb.keras import WandbCallback
import os
import time
import keras.backend as K


class cifar10vgg:

    def __init__(self, train=True):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32, 32, 3]
        self.batch_size = 128
        self.epoches = 120
        self.call_backs = self.get_callback_list()

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

        learning_rate = 0.1
        lr_drop = 20

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        callback_list = [
            ModelCheckpoint(filepath='cifar10vgg_asymmetric.h5',  # 目标文件的保存路径
                            monitor='val_accuracy',  # 监控验证损失
                            save_best_only=True),
            # EarlyStopping(monitor='val_accuracy', patience=20),
            ReduceLROnPlateau(monitor='val_accuracy'),
            TensorBoard("./logs/cifar10_no_residual")
            # WandbCallback()
        ]

        return callback_list

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
        tower_1 = DepthwiseConv2D((3, 3), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_1)

        # 第二路
        tower_2 = DepthwiseConv2D((3, 3), padding='same',
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
        tower_1 = DepthwiseConv2D((5, 1), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_1)
        tower_1 = DepthwiseConv2D((1, 5), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_1)

        # 第二路
        tower_2 = DepthwiseConv2D((5, 1), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_2)
        tower_2 = DepthwiseConv2D((1, 5), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_2)

        # 第三路
        tower_3 = DepthwiseConv2D((5, 1), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_3)
        tower_3 = DepthwiseConv2D((1, 5), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_3)

        # 第四路
        tower_4 = DepthwiseConv2D((5, 1), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_4)
        tower_4 = DepthwiseConv2D((1, 5), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_4)

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
        tower_1 = SeparableConv2D(interval / 4, (7, 1), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_1)
        tower_1 = SeparableConv2D(interval / 4, (1, 7), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_1)

        # 第二路
        tower_2 = SeparableConv2D(interval / 4, (7, 1), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_2)
        tower_2 = SeparableConv2D(interval / 4, (1, 7), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_2)

        # 第三路
        tower_3 = SeparableConv2D(interval / 4, (7, 1), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_3)
        tower_3 = SeparableConv2D(interval / 4, (1, 7), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_3)
        # 第四路
        tower_4 = SeparableConv2D(interval / 4, (7, 1), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_4)
        tower_4 = SeparableConv2D(interval / 4, (1, 7), padding='same',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_4)

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

        # inputs = Conv2D(out_filters, 1, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        # inputs = BatchNormalization()(inputs)
        #
        # x = keras.layers.add([inputs, x])
        # x = Activation('relu')(x)

        return x

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        weight_decay = 0.0005

        inputs = Input(shape=self.x_shape)

        x = Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        x = Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)

        x = self.py_block(x, 64, 128)
        x = self.py_block(x, 64, 128)
        x = self.py_block(x, 64, 128)
        x = MaxPooling2D()(x)

        x = self.py_block(x, 128, 256)
        # x = SpatialDropout2D(.1)(x)
        # x = Dropout(.3)(x)
        x = self.py_block(x, 128, 256)
        # x = SpatialDropout2D(.2)(x)
        # x = Dropout(.3)(x)
        x = self.py_block(x, 128, 256)

        x = GlobalMaxPooling2D()(x)

        outputs = Dense(self.num_classes, activation='softmax')(x)

        model_ = Model(inputs=inputs, outputs=outputs)

        model_.summary()

        return model_

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
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        # data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(datagen.flow(x_train, y_train,
                               batch_size=self.batch_size),
                  steps_per_epoch=x_train.shape[0] // self.batch_size,
                  epochs=self.epoches,
                  validation_data=(x_test, y_test),
                  validation_batch_size=128,
                  callbacks=self.call_backs)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = cifar10vgg()

    loss, acc = model.predict(x_test, y_test, batch_size=128)

    print(f"验证集：准确率:{acc}, 损失:{loss}")
