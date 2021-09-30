from tensorboard.backend.event_processing import event_accumulator
import os
import matplotlib.pyplot as plt
import random
import numpy as np

base_path = "logs\\"
temp_path = os.getcwd()


def get_file_path():
    dir_path = []
    for p in os.walk(base_path):
        if 'plugins' not in p[0]:

            if "train" in p[0] or "validation" in p[0]:
                dir_path.append(p[0])

    return dir_path


def split_train_valid(data):

    train = []
    valid = []

    for d in data:
        if "train" in d:
            train.append(d)
        else:
            valid.append(d)

    return train, valid


def file_split(_file_path):


    cifar10_train, cifar10_valid = split_train_valid(_file_path)
    # cifar100_train, cifar100_valid = split_train_valid(_cifar100)

    return cifar10_train, cifar10_valid


def show_train_history(data_list, train, dataset):
    """1、vgg_pyconv_light
       2、vgg
       3、vgg_pyconv
    """

    # 支持中文
    global vgg_acc, vgg_pyconv, vgg_pyconv_light_acc

    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'
    color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber, CB91_Purple, CB91_Violet]
    random.shuffle(color_list)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

    if dataset == 'cifar100':
        ticks = np.linspace(0, 0.7, 8)
    else:
        ticks = np.linspace(0, 1, 11)

    # 设置标题标注和字体大小
    plt.rcParams.update({"font.size": 20})  # 此处必须添加此句代码方可改变标题字体大小

    for i, p in enumerate(data_list):
        os.chdir(temp_path)
        os.chdir(p)

        file = os.listdir()[0]
        ea = event_accumulator.EventAccumulator(file)
        ea.Reload()
        print(ea.scalars.Keys())

        if i == 0:
            resnet_pyconv_light_acc = ea.scalars.Items('epoch_accuracy')
            resnet_pyconv_light_loss = ea.scalars.Items('epoch_loss')
        elif i == 1:
            resnet_pyconv_no_asy_acc = ea.scalars.Items('epoch_accuracy')
            resnet_pyconv_no_asy_loss = ea.scalars.Items('epoch_loss')
        else:
            # print(ea.scalars.Keys())
            resnet_pyconv_no_dsc_acc = ea.scalars.Items('epoch_accuracy')
            resnet_pyconv_no_dsc_loss = ea.scalars.Items('epoch_loss')

    fig = plt.figure(figsize=(6, 6), dpi=120)

    lw = 3

    ax1 = fig.add_subplot(111)
    ax1.plot([i.step for i in resnet_pyconv_light_acc], [i.value for i in resnet_pyconv_light_acc], label='pyconv_light_'+train+'_acc', lw=lw)
    # ax1.plot([i.step for i in resnet_pyconv_no_asy_acc], [i.value for i in resnet_pyconv_no_asy_acc], label='pyconv_no_asy_'+train+'_acc', lw=lw)
    ax1.plot([i.step for i in resnet_pyconv_no_dsc_acc], [i.value for i in resnet_pyconv_no_dsc_acc], label='pyconv_no_dsc_'+train+'_acc', lw=lw)
    ax1.set_xlabel("Epoch", fontsize=20)
    ax1.set_ylabel("Accuracy", fontsize=20)
    ax1.set_yticks(ticks)
    ax1.set_xlim(0)

    plt.title(dataset)
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()

    fig = plt.figure(figsize=(6, 6), dpi=120)

    lw = 3

    ax1 = fig.add_subplot(111)
    ax1.plot([i.step for i in resnet_pyconv_light_loss], [i.value for i in resnet_pyconv_light_loss], label='pyconv_light_'+train+'_loss', lw=lw)
    # ax1.plot([i.step for i in resnet_pyconv_no_asy_loss], [i.value for i in resnet_pyconv_no_asy_loss], label='pyconv_no_asy_'+train+'_loss', lw=lw)
    ax1.plot([i.step for i in resnet_pyconv_no_dsc_loss], [i.value for i in resnet_pyconv_no_dsc_loss], label='pyconv_no_dsc_'+train+'_loss', lw=lw)
    ax1.set_xlabel("Epoch", fontsize=20)
    ax1.set_ylabel("Loss", fontsize=20)
    ax1.set_xlim(0)

    plt.title(dataset)
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()


dir_path = get_file_path()
file_path = [x+"\\" for x in dir_path]

print(file_path)
cifar10_train, cifar10_valid= file_split(file_path)

print(cifar10_train)
print(cifar10_valid)

show_train_history(cifar10_train, "train", 'cifar10')
show_train_history(cifar10_valid, "valid", 'cifar10')










