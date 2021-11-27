#-*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.realpath(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
from layers import *
from dropout_layers import *
from bn_layers import *

class FullyConnectedNet(object):
    """
    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    """

    def __init__(self, input_dim=3*32*32,hidden_dims=[100],    num_classes=10,
                             dropout=0, use_batchnorm=False, reg=0.0,
                             weight_scale=1e-2, seed=None):
        """
        初始化全连接网络.    
        Inputs:
        - input_dim: 输入维度
        - hidden_dims: 隐藏层各层维度向量，如[100,100]
        - num_classes: 分类个数.
        - dropout: 如果dropout=0，表示不使用dropout.
        - use_batchnorm：布尔型，是否使用BN
        - reg:正则化衰减因子.
        - weight_scale:权重初始化范围，标准差.
        - seed: 使用seed产生相同的随机数。
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.params = {}

        ############################################################################
        #                             任务：初始化网络参数                                                                             #
        #                    权重参数初始化和前面章节类似                                                                        #
        #                    针对每一层神经元都要初始化对应的gamma和beta                                         #
        #                    如:第一层使用gamma1，beta1，第二层gamma2,beta2,                                 #
        #                     gamma初始化为1，beta初始化为0                                                                    # 
        ############################################################################
        layers_dims = [input_dim]+hidden_dims+[num_classes]
        for i in range(self.num_layers):
                self.params['W'+str(i+1)] = weight_scale*np.random.randn(layers_dims[i],
                                                                                                layers_dims[i+1])
                self.params['b'+str(i+1)] = np.zeros((1, layers_dims[i+1]))
                # 批量正则化
                if self.use_batchnorm and i < len(hidden_dims):
                        self.params['gamma'+str(i+1)] = np.ones((1, layers_dims[i+1]))
                        self.params['beta'+str(i+1)] = np.zeros((1, layers_dims[i+1]))
        ############################################################################
        #                                                        结束编码                                                                            #
        ############################################################################
        # dropout相关配置
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed     
        # 批量正则化
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        

    
    def loss(self, X, y=None):
        '''
        计算损失值

        Parameters
        ----------
        X : 训练数据
        y : 标签

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        # 设置执行模式
        mode = 'test' if y is None else 'train'

        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode     
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
        ############################################################################
        #                                    任务：执行全连接网络的前馈过程。                                                #
        #                             计算数据的分类得分，将结果保存在scores中。                                 #
        #            当使用dropout时，你需要使用self.dropout_param进行dropout前馈。            #
        #            当使用BN时，self.bn_params[0]传到第一层，self.bn_params[1]第二层        #
        ############################################################################
        outs, cache = {}, {}
        outs[0] = X
        num_h = self.num_layers-1
        for i in range(num_h):
            if self.use_dropout:
                outs[i+1], cache[i+1] = affine_relu_dropout_forward(
                    outs[i], self.params['W'+str(i+1)], self.params['b'+str(i+1)],
                    self.dropout_param)
            elif self.use_batchnorm:
                gamma = self.params['gamma'+str(i+1)]
                beta = self.params['beta'+str(i+1)]
                outs[i+1], cache[i+1] = affine_bn_relu_forward(outs[i],
                    self.params['W'+str(i+1)], self.params['b'+str(i+1)], gamma,
                    beta, self.bn_params[i])
            else:
                outs[i+1], cache[i+1] = affine_relu_forward(outs[i],
                    self.params['W'+str(i+1)], self.params['b'+str(i+1)])
        scores, cache[num_h+1] = affine_forward(outs[num_h], 
                        self.params['W'+str(num_h+1)], self.params['b'+str(num_h+1)])
        ############################################################################
        #                                                         结束编码                                                                         #
        ############################################################################
        # 测试模式
        if mode == 'test':
            return scores
        # 损失值与梯度
        loss, grads = 0.0, {}
        ############################################################################
        #                任务：实现全连接网络的反向传播。                                                                    #
        #                将损失值储存在loss中，梯度值储存在grads字典中                                         #
        #                当使用dropout时，需要求解dropout梯度                                                            #
        #                当使用BN时，需要求解BN梯度                                                                                #
        ############################################################################
        dout = {}
        loss, dy = softmax_loss(scores, y)
        h = self.num_layers-1
        for i in range(self.num_layers):
            loss += 0.5*self.reg*(np.sum(self.params['W'+str(i+1)]*self.params['W'+str(i+1)]))
        dout[h], grads['W'+str(h+1)], grads['b'+str(h+1)] = affine_backward(dy, cache[h+1])
        grads['W'+str(h+1)] += self.reg*self.params['W'+str(h+1)]
        for i in range(h):
            if self.use_dropout:
                dx, dw, db = affine_relu_dropout_backward(dout[h-i], cache[h-i])
                dout[h-1-i] = dx
                grads['W'+str(h-i)] = dw
                grads['b'+str(h-i)] = db
            elif self.use_batchnorm:
                dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dout[h-i], cache[h-i])
                dout[h-1-i] = dx
                grads['W'+str(h-i)] = dw
                grads['b'+str(h-i)] = db
                grads['gamma'+str(h-i)] = dgamma
                grads['beta'+str(h-i)] = dbeta
            else:
                dx, dw, db = affine_relu_backward(dout[h-i], cache[h-i])
                dout[h-1-i] = dx
                grads['W'+str(h-i)] = dw
                grads['b'+str(h-i)] = db
            grads['W'+str(h-i)] += self.reg*self.params['W'+str(h-i)]
        ############################################################################
        #                                                         结束编码                                                                         #
        ############################################################################
        return loss, grads
