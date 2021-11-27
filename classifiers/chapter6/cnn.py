#-*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.realpath(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
from cnn_layers import *

class ThreeLayerConvNet(object):
    """ 
    conv - relu - 2x2 max pool - affine - relu - affine - softmax
    """
    
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                             hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,):
        """
        初始化网络.
        
        Inputs:
        - input_dim: 输入数据形状 (C, H, W)
        - num_filters: 卷积核个数
        - filter_size: 卷积核尺寸
        - hidden_dim: 全连接层隐藏层个数
        - num_classes: 分类个数
        - weight_scale: 权重规模（标准差）
        - reg:权重衰减因子
        """
        self.params = {}
        self.reg = reg
        ############################################################################
        #                            任务：初始化权重参数                             #
        # 'W1'为卷积层参数，形状为(num_filters,C,filter_size,filter_size)             #
        # 'W2'为卷积层到全连接层参数，形状为((H/2)*(W/2)*num_filters, hidden_dim)       #
        #         'W3'隐藏层到全连接层参数                                            #
        ############################################################################
        C, H, W = input_dim
        self.params['W1'] = weight_scale*np.random.randn(num_filters, C, 
                    filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale*np.random.randn(int((H/2)*(W/2)*num_filters), 
                    hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale*np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             结束编码                                      #
        ############################################################################
         
 
    def loss(self, X, y=None):
        
        # 初始化参数
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        # 使用卷积层
        filter_size = W1.shape[2]
        # 设置卷积层和池化层所需要的参数
        conv_param = {'stride': 1, 'pad': int((filter_size - 1) / 2)}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        #                          任务： 实现前向传播                                #
        #                    计算每类得分，将其存放在scores中                          #
        ############################################################################
        # 组合卷积层：卷积，ReLU，池化
        conv_forward_out_1, cache_forward_1 = conv_relu_pool_forward(X, 
                self.params['W1'], self.params['b1'], conv_param, pool_param)
        # affine层
        affine_forward_out_2, cache_forward_2 = affine_forward(conv_forward_out_1,
                self.params['W2'], self.params['b2'])
        # relu层
        affine_relu_2, cache_relu_2 = relu_forward(affine_forward_out_2)
        # affine层
        scores, cache_forward_3 = affine_forward(affine_relu_2, self.params['W3'],
                self.params['b3'])
        ############################################################################
        #                           结束编码                                        #
        ############################################################################
        if y is None:
            return scores
            
        loss, grads = 0, {}
        ############################################################################
        #                        任务：实现反向转播                                   #
        #                      注意：别忘了权重衰减项                                 #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        loss += self.reg*0.5*(np.sum(self.params['W1']**2)
                              +np.sum(self.params['W2']**2)
                              +np.sum(self.params['W3']**2))
        dX3, grads['W3'], grads['b3'] = affine_backward(dout, cache_forward_3)
        dX2 = relu_backward(dX3, cache_relu_2)
        dX2, grads['W2'], grads['b2'] = affine_backward(dX2, cache_forward_2)
        dX1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dX2, cache_forward_1)
        grads['W3'] = grads['W3']+self.reg*self.params['W3']
        grads['W2'] = grads['W2']+self.reg*self.params['W2']
        grads['W1'] = grads['W1']+self.reg*self.params['W1']
        ############################################################################
        #                          结束编码                                        #
        ############################################################################
        return loss, grads
    
