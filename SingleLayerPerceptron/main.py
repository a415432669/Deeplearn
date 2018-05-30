# coding:utf-8
import math


def sigmoid(x):
    return 1.0/(1+math.exp(-x))


class Record:# 输入的一条训练数据
    def __init__(self):
        feature_vector = []# 特征向量
        label = None# 标签
        return


class Node: # 神经元节点
    def __init__(self):
        self.input_list = []# 输入的神经元列表
        self.activated = False# 这个神经元是否已经计算出输出信号
        self.recent_output = None# 上一次的输出信号
        self.threshold = 0.0# 阈值θ
        self.activation_func = lambda s: 1.0 / (1 + math.exp(-s))  # default func: sigmoid  function
        return

    def add_input(self, node):# 添加输入结点
        self.input_list.append([node, 1.0])# [结点对象，权重w] 权重默认为1
        return

    def set_threshold(self, th):# 设置阈值θ
        self.threshold = th
        return

    def output(self):# 通过激活函数计算输出信号
        sum_ = 0.0
        for p in self.input_list:
            prev_node = p[0]
            sum_ += prev_node.output() * p[1]
        self.recent_output = self.activation_func(sum_ - self.threshold)
        return self.recent_output


class InputNode(Node):# 神经元输入节点
    def __init__(self):
        Node.__init__(self)
        self.activation_func = lambda s: s# 注意激活函数是 f(x)=x 
        return

    def set_input_val(self, val):
        Node.set_threshold(self, -val)# 输入的结点列表为空，设置阈值为 -val
        return


class OutputNode(Node):# 神经元输出节点
    def __init__(self):
        Node.__init__(self)
        self.threshold = 4.0
        return


class NeuralNetwork:# 抽象神经网络
    def __init__(self):
        self.eta = 0.5# 学习率η
        self.data_set = []# 输入的数据集，列表内的元素为 Record对象

    def set_data_set(self, data_set_):# 设置数据集
        self.data_set = data_set_
        return


class SingleLayerNeuralNetwork(NeuralNetwork):# 感知机模型
    def __init__(self):
        NeuralNetwork.__init__(self)
        self.perceptron = OutputNode()# 一个输出结点
        self.input_node_list = []
        self.perceptron.threshold = 4.0
        return

    def add_input_node(self):# 添加输入结点
        inode = InputNode()
        self.input_node_list.append(inode)
        self.perceptron.add_input(inode)
        return

    def set_input(self, value_list):# 给每个输入结点设置输入数据
        assert len(value_list) == len(self.input_node_list)
        for index in range(0, len(value_list), 1):
            value = value_list[index]
            node = self.input_node_list[index]
            node.set_input_val(value)
        return

    def adjust(self, label):# 每条记录训练后，自动调整内部参数w
        for prev_node_pair in self.perceptron.input_list:
            delta_weight = self.eta * (label - self.perceptron.recent_output) * prev_node_pair[0].recent_output
            origin_weight = prev_node_pair[1]
            prev_node_pair[1] = origin_weight + delta_weight
        return

    def run(self):# 开始跑训练集
        for data in self.data_set:# 遍历训练集
            self.set_input(data.feature_vector)
            record_ = self.perceptron.output()
            print (record_)
            self.adjust(data.label)# 每条记录训练后，自动调整内部参数w
        return

if __name__ == '__main__':
    file_handler = open('./train.txt')
    data_set = []
    line = file_handler.readline()
    while line:# 遍历文件里的数据
        record = Record()
        item_feature_vector = []
        str_list = line.split()
        item_feature_vector.append(float(str_list[0]))
        item_feature_vector.append(float(str_list[1]))

        record.feature_vector = item_feature_vector
        record.label = float(str_list[2])
        data_set.append(record)
        line = file_handler.readline()
    print (len(data_set))

    ann = SingleLayerNeuralNetwork()# 实例化感知机对象
    ann.add_input_node()
    ann.add_input_node()# 添加两个输入结点
    ann.set_data_set(data_set)# 设置数据集
    ann.run()# 等待结果