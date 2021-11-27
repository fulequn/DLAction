#-*- coding: utf-8 -*-
import numpy as np
from .layers import *


class FullyConnectedNet(object):
    """
    深层全连接神经网络，其中隐藏层使用ReLU作为激活函数，输出层使用softmax作为分类器
    该网络结构应该为     {affine - relu}x(L -1) -affine - softmax
    """
    def __init__(self, input_dim=3*32*32, hidden_dims=[50,50], num_classes=10,
                 reg=0.0, weight_scale=1e-3):
        """
        初始化网络.

        Inputs:
        - input_dim: 输入数据维度
        - hidden_dim: 隐藏层各层维度
        - num_classes: 分类数量
        - weight_scale: 权重范围，给予初始化权重的标准差
        - reg: L2正则化的权重衰减系数.
        """
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.params = {}
        # 这里存储的是每层的神经元数量。 
        layers_dims = [input_dim] + hidden_dims + [num_classes] 
        ############################################################################
        #                    任务：初始化任意多层权重以及偏置项                    #
        #      权重应该服从标准差为weight_scale的高斯分布，偏置项应该初始化为0,    #
        #        所有权重矩阵和偏置向量应该存放在self.params字典中。               #
        #   第一层的权重和偏置使用键值 'W1'以及'b1'，第n层使用'Wn'以及'bn'         #
        ############################################################################   
        for i in range(self.num_layers):
            self.params['W'+str(i+1)] = weight_scale*np.random.randn(
                layers_dims[i], layers_dims[i+1])
            self.params['b'+str(i+1)] = np.zeros((1, layers_dims[i+1]))
        ############################################################################
        #                           结束编码                                       #
        ############################################################################


    
    def loss(self, X, y=None):
        """
        计算数据X的损失值以及梯度.

        Inputs:
        - X: 输入数据，形状为(N, d_1, ..., d_k)的numpy数组。
        - y: 数据类标，形状为(N,)的numpy数组。

        Returns:
        如果y为 None, 表明网络处于测试阶段直接返回输出层的得分即可:
        - scores:形状为 (N, C)，其中scores[i, c] 是数据 X[i] 在第c类上的得分.
        
        如果y为 not None, 表明网络处于训练阶段，返回一个元组:
        - loss:数据的损失值
        - grads: 与参数字典相同的梯度字典，键值和参数字典的键值要相同
        """  
        scores = None
        ############################################################################
        #              任务: 实现深层网络的前向传播过程，                          #
        #                       计算各数据的分类得分                               #
        ############################################################################
        cache_relu, outs, cache_out = {}, {}, {}
        outs[0] = X
        num_h = self.num_layers-1
        for i in range(num_h):
            outs[i+1], cache_relu[i+1] = affine_relu_forward(
                outs[i], self.params['W'+str(i+1)], self.params['b'+str(i+1)])
        scores, cache_out = affine_forward(outs[num_h], 
                                           self.params['W'+str(num_h+1)],
                                           self.params['b'+str(num_h+1)])
        ############################################################################
        #                            结束编码                                      #
        ############################################################################
        loss, grads = 0.0, {}
        ############################################################################
        #                    任务：实现深层网络的反向传播过程，                    #
        #            将损失值存储在loss中,将各层梯度储存在grads字典中。            #
        #                           注意：别忘了还要计算权重衰减哟。               #
        ############################################################################
        dout, daffine = {}, {}
        loss, dy = softmax_loss(scores, y)
        # 隐藏层数
        h = self.num_layers-1
        # 最后一层是affine层，前面的均是组合层
        for i in range(self.num_layers):
            loss += 0.5*self.reg*(np.sum(self.params['W'+str(i+1)]*self.params['W'+str(i+1)]))
            dout[h], grads['W'+str(h+1)], grads['b'+str(h+1)] = affine_backward(dy, cache_out)
            grads['W'+str(h+1)] += self.reg*self.params['W'+str(h+1)]
            for i in range(h):
                dout[h-i-1], grads['W'+str(h-i)], grads['b'+str(h-i)] = affine_relu_backward(
                    dout[h-i], cache_relu[h-i])
                grads['W'+str(h-i)] += self.reg*self.params['W'+str(h-i)]
        ############################################################################
        #                             结束编码                                     #
        ############################################################################

        return loss, grads

    def train(self, X, y, X_val, 
              y_val,learning_rate=1e-3, learning_rate_decay=0.95,
              num_iters=100,batch_size=200, verbose=False):
        """
        使用随机梯度下降训练神经网络
        Inputs:
        - X: 训练数据
        - y: 训练类标.
        - X_val: 验证数据.
        - y_val:验证类标.
        - learning_rate: 学习率.
        - learning_rate_decay: 学习率衰减系数
        - reg: 权重衰减系数.
        - num_iters: 迭代次数.
        - batch_size: 批量大小.
        - verbose:是否在训练过程中打印结果.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []
        best_val=-1
        for it in range(num_iters):

            X_batch = None
            y_batch = None
            
            sample_index = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[sample_index, :]  # (batch_size,D)
            y_batch = y[sample_index]  # (1,batch_size)

            #计算损失以及梯度
            loss, grads = self.loss(X_batch, y=y_batch)
            loss_history.append(loss)
            
            #修改权重
            ############################################################################
            #                    任务：修改深层网络的权重                              #
            ############################################################################
            for i,j in self.params.items():
                self.params[i] += -learning_rate*grads[i]
            ############################################################################
            #                              结束编码                                    #
            ############################################################################
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            if it % iterations_per_epoch == 0:
                # 检验精度
                train_acc = (self.predict(X) == y).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                # 记录数据
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                if (best_val < val_acc):
                    best_val = val_acc
                # 学习率衰减
                learning_rate *= learning_rate_decay
        # 以字典的形式返回结果
        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
          'best_val_acc':best_val
        }
    
    def predict(self, X):
        """
        Inputs:
        - X: 输入数据
        Returns:
        - y_pred: 预测类别
        """
        y_pred = None
        ###########################################################################
        #                   任务： 执行深层网络的前向传播，                       #
        #                  然后使用输出层得分函数预测数据类标                     #
        ###########################################################################
        outs = {}
        outs[0] = X
        num_h = self.num_layers-1
        for i in range(num_h):
            outs[i+1], _ =affine_relu_forward(outs[i], self.params['W'+str(i+1)],
                                              self.params['b'+str(i+1)])
        scores, _ = affine_forward(outs[num_h], self.params['W'+str(num_h+1)], 
                                   self.params['b'+str(num_h+1)])
        y_pred = np.argmax(scores, axis=1)
        ###########################################################################
        #                             结束编码                                    #
        ###########################################################################
        return y_pred
