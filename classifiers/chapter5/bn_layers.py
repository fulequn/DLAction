#-*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.realpath(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
from layers import *
from dropout_layers import *

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    使用类似动量衰减的运行时平均，计算总体均值与方差 例如:
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    Input:
    - x: 数据(N, D)
    - gamma: 缩放参数 (D,)
    - beta: 平移参数 (D,)
    - bn_param: 字典型，使用下列键值:
        - mode: 'train' 或'test'; 
        - eps: 保证数值稳定
        - momentum: 运行时平均衰减因子 
        - running_mean: 形状为(D,)的运行时均值
        - running_var : 形状为 (D,)的运行时方差

    Returns 元组:
    - out: 输出(N, D)
    - cache: 用于反向传播的缓存
    """
    # 不同阶段
    mode = bn_param['mode']
    # 防止为0
    eps = bn_param.get('eps', 1e-5)
    # 动量衰减
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape    # N:数据数目 D:数据维度
    # 获取运行时均差以及运行时方差，默认为0
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #############################################################################
        #                            任务：实现训练阶段BN的前向传播                                                             #
        #            首先，你需要计算输入数据的均值和方差 ；                                                            #
        #            然后，使用均值和方差将数据进行归一化处理；                                                     #
        #            之后，使用gamma和beta参数将数据进行缩放和平移；                                            #
        #            最后，将该批数据均值和方差添加到累积均值和方差中；                                     #
        #            注意：将反向传播时所需的所有中间值保存在cache中。                                        #
        #############################################################################
        mu = 1/float(N)*np.sum(x, axis=0)
        xmu = x-mu
        carre = xmu**2
        var = 1/float(N)*np.sum(carre, axis=0)
        sqrtvar = np.sqrt(var+eps)
        invvar = 1./sqrtvar
        va2 = xmu*invvar
        va3 = gamma*va2
        out = va3+beta
        running_mean = momentum*running_mean+(1.0-momentum)*mu
        running_var = momentum * running_var+(1.0-momentum)*var
        cache=(mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param)
        #############################################################################
        #                                                         结束编码                                                                         #
        #############################################################################
    elif mode == 'test':
        #############################################################################
        #                    任务：实现测试阶段BN的前向传播                                                                     #
        #            首先，使用运行时均值与方差归一化数据，                                                             #
        #            然后，使用gamma和beta参数缩放，平移数据。                                                        #
        #############################################################################
        mu = running_mean
        var = running_var
        xhat = (x-mu)/np.sqrt(var+eps)
        out = gamma*xhat+beta
        cache = (mu, var, gamma, beta, bn_param)
        #############################################################################
        #                                                         结束编码                                                                         #
        #############################################################################
    else:
        raise ValueError('无法识别的BN模式： "%s"' % mode)
 
     # 更新运行时均值，方差
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    BN反向传播 
    Inputs:
    - dout: 上层梯度 (N, D)
    - cache: 前向传播时的缓存.
    
    Returns 元组:
    - dx: 数据梯度 (N, D)
    - dgamma: gamma梯度 (D,)
    - dbeta: beta梯度 (D,)
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    #                                任务：实现BN反向传播                                                                             #
    #                     将结果分别保存在dx,dgamma,dbeta中                                                             #
    #############################################################################
    mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param=cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape
    # 第9步反向传播
    dva3 = dout
    dbeta = np.sum(dout, axis=0)
    # 第8步反向传播
    dva2 = gamma*dva3
    dgamma = np.sum(va2*dva3, axis=0)    
    # 第7步反向传播
    dxmu = invvar*dva2
    dinvvar = np.sum(xmu*dva2, axis=0)
    # 第6步反向传播
    dsqrtvar = -1./(sqrtvar**2)*dinvvar
    # 第5步反向传播
    dvar = 0.5*(var+eps)**(-0.5)*dsqrtvar
    # 第4步反向传播
    dcarre = 1/float(N)*np.ones((carre.shape))*dvar
    # 第3步反向传播
    dxmu += 2*xmu*dcarre
    # 第2步反向传播
    dx = dxmu
    dmu = -np.sum(dxmu, axis=0)
    # 第1步反向传播
    dx += 1/float(N)*np.ones((dxmu.shape))*dmu
    #############################################################################
    #                                                         结束编码                                                                            #
    #############################################################################
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    可选的BN反向传播
    """
    dx, dgamma, dbeta = None, None, None
    mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum((x - mu) * (var + eps)**(-1. / 2.) * dout, axis=0)
    dx = (1./N) * gamma * (var + eps)**(-1./2.)*(N*dout-np.sum(
                        dout, axis=0)-(x-mu)*(var+eps)**(-1.0)*np.sum(dout*(x-mu),axis=0))
 
    return dx, dgamma, dbeta


def affine_bn_relu_forward(x,w,b,gamma, beta,bn_param):
    x_affine,cache_affine= affine_forward(x,w,b)
    x_bn,cache_bn = batchnorm_forward(x_affine,gamma, beta,bn_param)
    out,cache_relu = relu_forward(x_bn)
    cache = (cache_affine,cache_bn,cache_relu)
    return out,cache

def affine_bn_relu_backward(dout,cache):
    cache_affine,cache_bn,cache_relu = cache
    drelu = relu_backward(dout,cache_relu)
    dbn,dgamma, dbeta= batchnorm_backward_alt(drelu,cache_bn)
    dx,dw,db = affine_backward(dbn,cache_affine)
    return dx,dw,db,dgamma,dbeta

