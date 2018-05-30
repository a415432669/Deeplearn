import math
def sigmoid(x):
    return 1.0/(1+math.exp(-x))

def sign(x):
    if x>=0 :
        return 1
    if x<0 : 
        return 0

class Record:#输入一条的训练数据集
    def __init__(self):
        feature_vector = []#特征向量
        label = None #标签
        return
    def __str__(self):
        return str(self.__dict__)
    def __repr__(self):
        return self.__str__()

class Node: #神经元节点
    def __init__(self):
        self.input_list = [] #输入的神经元列表
        self.activated = False #这个神经元是否已经计算出输出信号
        self.recent_output = None #上一次输出的信号
        self.threshold = 0.0 #阈值
        self.activation_func = sign #默认的激活函数
        return
    
    def add_input(self,node):#添加输入节点
        self.input_list.append([node,1.0]) #[节点对象，权重w]权重默认为1
        return
    
    def set_threshold(self,th): #设置阈值
        self.threshold = th
        return

    def output(self): #通过激活函数计算输出信号
        sum_ = 0.0
        # print(self.input_list)
        for p in self.input_list:
            prev_node = p[0] #输入的节点
            sum_ += prev_node.output()*p[1] #输入节点的输出*权重w
        self.recent_output = self.activation_func(sum_ - self.threshold)
        return self.recent_output
    def __str__(self):
        return str(self.__dict__)
    def __repr__(self):
        return self.__str__()
        
class InputNode(Node):#神经元输入节点
    def __init__(self):
        Node.__init__(self)
        self.activation_func = lambda s:s  #注意激活函数是f(x) = x
        return

    def set_input_val(self,val):
        Node.set_threshold(self,-val)#输入的节点列表为空，设置阈值为-val
        return

class OutputNode(Node):#神经元输出节点
    def __init__(self):
        Node.__init__(self)
        self.threshold = 4.0
        return

class NeuralNetwork:#抽象神经网络
    def __init__(self):
        self.eta = 0.5 #学习率
        self.data_set = [] #输入的数据集，列表内的元素为Record对象

    def set_data_set(self,data_set_):#设置数据集
        self.data_set = data_set_
        return
    def __str__(self):
        return str(self.__dict__)
    def __repr__(self):
        return self.__str__()

class SingleLayerNeuralNetwork(NeuralNetwork):#感知机模型
    def __init__(self):
        NeuralNetwork.__init__(self)
        self.perceptron = OutputNode() #一个输出节点
        self.input_node_list = []
        self.perceptron.threshold = 4.0
        return 
    
    def add_input_node(self): #添加输入节点
        inode = InputNode() #创建输入节点
        self.input_node_list.append(inode) #添加新创建的输入节点至输入节点列表
        self.perceptron.add_input(inode) #添加新创建的输入节点至输出节点
        return

    def set_input(self,value_list): #给每个输入节点设置输入数据
        assert len(value_list) == len(self.input_node_list)
        for index in range(0,len(value_list),1):
            value = value_list[index]
            node = self.input_node_list[index]
            node.set_input_val(value)
        return

    def adjust(self,label): #每条记录训练后，自动调整内部参数w
        for prev_node_pair in self.perceptron.input_list:
            delta_weight = self.eta * (label - self.perceptron.recent_output) * prev_node_pair[0].recent_output
            origin_weight = prev_node_pair[1] 
            prev_node_pair[1] = origin_weight + delta_weight
        return

    def run(self): #开始跑训练集
        for data in self.data_set: #遍历训练集
            self.set_input(data.feature_vector)

            # print(self.perceptron)
            record_ = self.perceptron.output()
            print('单次结果输出：%s'%record_) 
            
            self.adjust(data.label) #每条记录训练集后，自动调整内部参数w
            print(self.perceptron.input_list[0][1],self.perceptron.input_list[1][1])
        return


if __name__ == '__main__':
    data_set = []
    with open('./train.txt') as f:
        line = f.readline()
        while line:#遍历文件里的数据
            record = Record()
            
            item_feature_vector = []
            str_list = line.split()
            item_feature_vector.append(float(str_list[0]))
            item_feature_vector.append(float(str_list[1]))

            record.feature_vector = item_feature_vector
            record.label = float(str_list[2])
            data_set.append(record)
            # print(record)
            line = f.readline()
        print('数据集长度：%s'%len(data_set))
        print(data_set[0])

    ann = SingleLayerNeuralNetwork() #实例化感知机对象
    ann.add_input_node()
    ann.add_input_node() #添加两个输入节点
    ann.set_data_set(data_set) #设置数据集
    # print(ann)
    ann.run() #等待结果

