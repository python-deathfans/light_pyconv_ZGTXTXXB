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

    _cifar100 = [x for x in file_path if "cifar100" in x]
    _cifar10 = []

    for x in _file_path:
        if x not in _cifar100:
            _cifar10.append(x)

    cifar10_train, cifar10_valid = split_train_valid(_cifar10)
    cifar100_train, cifar100_valid = split_train_valid(_cifar100)

    return cifar10_train, cifar10_valid, cifar100_train, cifar100_valid


def show_train_history(data_list, train, dataset):
    """1、vgg_pyconv_light
       2、vgg
       3、vgg_pyconv
    """

    # 支持中文
    global vgg_acc, vgg_pyconv, vgg_pyconv_light_acc

    # os.chdir("img")

    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'
    color_list = [CB91_Blue, CB91_Pink, CB91_Green]
    # random.shuffle(color_list)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

    if dataset == 'CIFAR-100' and train == "valid":
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
            ligt_acc = ea.scalars.Items('epoch_accuracy')
            light_loss = ea.scalars.Items('epoch_loss')
        elif i == 1:
            no_asymmetric_acc = ea.scalars.Items('epoch_accuracy')
            no_asymmetric_loss = ea.scalars.Items('epoch_loss')
        else:
            # print(ea.scalars.Keys())
            no_dsc_acc = ea.scalars.Items('epoch_accuracy')
            no_dsc_loss = ea.scalars.Items('epoch_loss')

    fig = plt.figure(figsize=(6, 6), dpi=120)

    ax1 = fig.add_subplot(111)
    ax1.plot([i.step for i in ligt_acc], [i.value for i in ligt_acc], label='light', lw=3)
    # ax1.set_xlim(0)
    ax1.plot([i.step for i in no_asymmetric_acc], [i.value for i in no_asymmetric_acc], label='no_asymmetric', lw=3)
    # ax1.plot([i.step for i in no_dsc_acc], [i.value for i in no_dsc_acc], label='no_dwc', lw=3)
    ax1.set_xlabel("Epoch", fontsize=20)
    ax1.set_ylabel("Accuracy", fontsize=20)
    ax1.set_yticks(ticks)
    ax1.set_xlim(0)

    plt.title(dataset)
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()

    fig = plt.figure(figsize=(6, 6), dpi=120)

    ax1 = fig.add_subplot(111)
    ax1.plot([i.step for i in light_loss], [i.value for i in light_loss], label='light', lw=3)
    ax1.plot([i.step for i in no_asymmetric_loss], [i.value for i in no_asymmetric_loss],  label='no_asymmetric', lw=3)
    # ax1.plot([i.step for i in no_dsc_loss], [i.value for i in no_dsc_loss],label='no_dwc', lw=3)
    ax1.set_xlabel("Epoch", fontsize=20)
    ax1.set_ylabel("Loss", fontsize=20)
    ax1.set_xlim(0)
    # ax1.set_ylim(top=1.2)

    plt.title(dataset)
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()


dir_path = get_file_path()
file_path = [x+"\\" for x in dir_path]

cifar10_train, cifar10_valid, cifar100_train, cifar100_valid = file_split(file_path)

print(cifar10_train)
print(cifar10_valid)
print(cifar100_train)
print(cifar100_valid)

# show_train_history(cifar10_train, "train", 'cifar10')
show_train_history(cifar10_valid, "valid", 'CIFAR-10')
# show_train_history(cifar100_train, "train", 'cifar100')
show_train_history(cifar100_valid, "valid", 'CIFAR-100')










