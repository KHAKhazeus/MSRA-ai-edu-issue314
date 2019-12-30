# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet_4_1 import *
from MiniFramework.ActivationLayer import *
from ExtendedDataReader.MnistImageDataReader import *
import threading

lock = threading.Lock()

def LoadData():
    #读取时最好上锁
    lock.acquire()
    print("reading MNIST data...")
    dr = MnistImageDataReader(mode="vector")
    dr.ReadData()
    dr.NormalizeX()
    dr.NormalizeY(NetType.MultipleClassifier)
    dr.GenerateValidationSet(k=20)
    print(dr.num_validation, dr.num_example, dr.num_test, dr.num_train)
    #释放锁
    lock.release()
    return dr

#训练线程
class TrainThread(threading.Thread):
    def __init__(self, mnist_comparenet):
        super(TrainThread, self).__init__()
        self.mnist_comparenet = mnist_comparenet
    def run(self):
        self.mnist_comparenet.train()

#包装测试网络
class MNISTCompareNet():
    def __init__(self, init_method, activator, stop_thresh=0.12):
        #固定超参数
        self.dataReader = LoadData()
        self.num_input = self.dataReader.num_feature
        self.num_hidden1 = 128
        self.num_hidden2 = 64
        self.num_hidden3 = 32
        self.num_hidden4 = 16
        self.num_output = 10
        self.max_epoch = 20
        self.batch_size = 64
        self.learning_rate = 0.1
        params = HyperParameters_4_1(
            self.learning_rate, self.max_epoch, self.batch_size,
            net_type=NetType.MultipleClassifier,
            init_method=init_method,
            stopper=Stopper(StopCondition.Nothing, 0))
        net = NeuralNet_4_1(params, "MNIST")
        fc1 = FcLayer_1_1(self.num_input, self.num_hidden1, params)
        net.add_layer(fc1, "fc1")
        r1 = ActivationLayer(Relu())
        net.add_layer(r1, "r1")
        
        fc2 = FcLayer_1_1(self.num_hidden1, self.num_hidden2, params)
        net.add_layer(fc2, "fc2")
        r2 = ActivationLayer(Relu())
        net.add_layer(r2, "r2")

        fc3 = FcLayer_1_1(self.num_hidden2, self.num_hidden3, params)
        net.add_layer(fc3, "fc3")
        r3 = ActivationLayer(Relu())
        net.add_layer(r3, "r3")

        fc4 = FcLayer_1_1(self.num_hidden3, self.num_hidden4, params)
        net.add_layer(fc4, "fc4")
        r4 = ActivationLayer(Relu())
        net.add_layer(r4, "r4")

        fc5 = FcLayer_1_1(self.num_hidden4, self.num_output, params)
        net.add_layer(fc5, "fc5")
        softmax = ClassificationLayer(Softmax())
        net.add_layer(softmax, "softmax")
        self.net = net

    #训练
    def train(self):
        self.net.train(self.dataReader, checkpoint=0.05, need_test=True)
    
    #展示结果
    def ShowResult(self):
        self.net.ShowLossHistory(xcoord=XCoordinate.Iteration)

def net(init_method, activator):

    max_epoch = 1
    batch_size = 5
    learning_rate = 0.02

    params = HyperParameters_4_1(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.Fitting,
        init_method=init_method)

    net = NeuralNet_4_1(params, "level1")
    num_hidden = [128,128,128,128,128,128,128]
    fc_count = len(num_hidden)-1
    layers = []

    for i in range(fc_count):
        fc = FcLayer_1_1(num_hidden[i], num_hidden[i+1], params)
        net.add_layer(fc, "fc")
        layers.append(fc)

        ac = ActivationLayer(activator)
        net.add_layer(ac, "activator")
        layers.append(ac)
    # end for
    
    # 从正态分布中取1000个样本，每个样本有num_hidden[0]个特征值
    # 转置是为了可以和w1做矩阵乘法
    x = np.random.randn(1000, num_hidden[0])

    # 激活函数输出值矩阵列表
    a_value = []

    # 依次做所有层的前向计算
    input = x
    for i in range(len(layers)):
        output = layers[i].forward(input)
        # 但是只记录激活层的输出
        if isinstance(layers[i], ActivationLayer):
            a_value.append(output)
        # end if
        input = output
    # end for

    for i in range(len(a_value)):
        ax = plt.subplot(1, fc_count+1, i+1)
        ax.set_title("layer" + str(i+1))
        plt.ylim(0,10000)
        if i > 0:
            plt.yticks([])
        ax.hist(a_value[i].flatten(), bins=25, range=[0,1])
    #end for
    # super title
    plt.suptitle(init_method.name + " : " + activator.get_name())
    plt.show()

if __name__ == '__main__':
    net(InitialMethod.Normal, Sigmoid())
    net(InitialMethod.Xavier, Sigmoid())
    net(InitialMethod.Xavier, Relu())
    net(InitialMethod.MSRA, Relu())

    #MNIST参数组合测试
    normal_tan = MNISTCompareNet(InitialMethod.Normal, Tanh())
    normal_relu = MNISTCompareNet(InitialMethod.Normal, Relu())
    normal_sigmoid = MNISTCompareNet(InitialMethod.Normal, Sigmoid())
    xavier_tan = MNISTCompareNet(InitialMethod.Xavier, Tanh())
    xavier_relu = MNISTCompareNet(InitialMethod.Xavier, Relu())
    xavier_sigmoid = MNISTCompareNet(InitialMethod.Xavier, Sigmoid())
    msra_tan = MNISTCompareNet(InitialMethod.MSRA, Tanh())
    msra_relu = MNISTCompareNet(InitialMethod.MSRA, Relu())
    msra_sigmoid = MNISTCompareNet(InitialMethod.MSRA, Sigmoid())
    #创建训练线程
    threads = [TrainThread(normal_tan),TrainThread(normal_relu), TrainThread(normal_sigmoid), 
        TrainThread(xavier_tan), TrainThread(xavier_relu), TrainThread(xavier_sigmoid), 
        TrainThread(msra_tan), TrainThread(msra_relu), TrainThread(msra_sigmoid)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    #按序展示结果
    normal_tan.ShowResult()
    normal_relu.ShowResult()
    normal_sigmoid.ShowResult()
    xavier_tan.ShowResult()
    xavier_relu.ShowResult()
    xavier_sigmoid.ShowResult()
    msra_tan.ShowResult()
    msra_relu.ShowResult()
    msra_sigmoid.ShowResult()