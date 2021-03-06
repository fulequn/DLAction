#-*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.realpath(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
from softmax_loss import *

class Softmax(object):
    
    def __init__(self):
        self.W = None
        
    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
        '''
        使用最小批量梯度下降算法训练Softmax分类器

        Parameters
        ----------
        X : TYPE
            数据
        y : TYPE
            数据类标
        learning_rate : TYPE, optional
            学习率. The default is 1e-3.
        reg : TYPE, optional
            权重衰减因子. The default is 1e-5.
        num_iters : TYPE, optional
            迭代次数. The default is 100.
        batch_size : TYPE, optional
            批大小. The default is 200.
        verbose : TYPE, optional
            是否显示中间过程. The default is False.

        Returns
        -------
        loss_history : TYPE
            DESCRIPTION.

        '''
        # 获得训练数据及维度
        num_train, dim = X.shape
        # 我们的计数是从0开始，因此10分类任务其y的最大值为9
        num_classes = np.max(y) + 1
        # 如果没有权重
        if self.W is None:
            # 随机初始化 W
            self.W = 0.001 * np.random.randn(dim, num_classes)
        # 储存每一轮的损失结果 W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None
            #########################################################################
            #                             任务:                                     #
            #     从训练数据 X 中采样大小为batch_size的数据及其类标，               #
            #     并将采样数据及其类标分别存储在X_batch，y_batch中                  #
            #        X_batch的形状为  (dim,batch_size)                              #
            #        y_batch的形状为  (batch_size)                                  #
            #     提示: 可以使用np.random.choice函数生成indices.                    #
            #            重复采样要比非重复采样快许多                               #
            #########################################################################
            # False表示不可以取相同数字
            indices = np.random.choice(num_train, batch_size, False)
            X_batch = X[indices, :]
            y_batch = y[indices]
            #########################################################################
            #                       结束编码                                        #
            #########################################################################
            # 计算损失及梯度
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            # 更新参数
            #########################################################################
            #                       任务:                                           #
            #               使用梯度及学习率更新权重                                #
            #########################################################################
            self.W = self.W - learning_rate*grad 
            #########################################################################
            #                      结束编码                                         #
            #########################################################################
            if verbose and it % 500 == 0:
                print('迭代次数 %d / %d: loss %f' % (it, num_iters, loss))
        return loss_history
    
    
    def predict(self, X):
        """
        使用已训练好的权重预测数据类标
        Inputs:
        - X:数据形状 (N,D) .表示N条数据，每条数据有D维
        Returns:
        - y_pred：形状为(N,) 数据X的预测类标，y_pred是一个长度维N的一维数组，
        每一个元素是预测的类标整数
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        #                         任务:                                           #
        #              执行预测类标任务，将结果储存在y_pred                       #
        ###########################################################################
        # 找到可能性最高的类别作为预测分类
        y_pred = np.argmax(X.dot(self.W), axis=1)	
        ###########################################################################
        #                           结束编码                                      #
        ###########################################################################
        return y_pred
    
    
    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative. 
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

