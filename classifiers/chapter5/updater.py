#-*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.realpath(os.path.dirname(os.path.realpath(__file__))))

import numpy as np

"""
频繁使用在训练神经网络中的一阶梯度更新规则。每次更新接受当前的权重，
对应的梯度，以及相关配置进行权重更新。
def update(w, dw, config=None):
Inputs:
    - w:当前权重.
    - dw: 和权重形状相同的梯度.
    - config: 字典型超参数配置，比如学习率，动量值等。如果更新规则需要用到缓存，
        在配置中需要保存相应的缓存。

Returns:
    - next_w: 更新后的权重.
    - config: 更新规则相应的配置.
"""


def sgd(w, dw, config=None):
    """
    随机梯度下降更新规则.

    config 格式:
    - learning_rate: 学习率.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config



def sgd_momentum(w, dw, config=None):
    """
    动量随机梯度下降更新规则。
    config 使用格式:
    - learning_rate: 学习率。
    - momentum: [0,1]的动量衰减因子，0表示不使用动量，即退化为SGD。
    - velocity: 和w，dw形状相同的速度。
    """
    if config is None: 
      config = { }
    # 设置默认值
    # 学习率
    config.setdefault('learning_rate', 1e-2)
    # 动量衰减因子
    config.setdefault('momentum', 0.9)
    # 速度
    v = config.setdefault('velocity', np.zeros_like(w))
    
    next_w = None
    #############################################################################
    #                                  任务：实现动量更新                                                                    #
    #                 更新后的速度存放在v中，更新后的权重存放在next_w中                                 #
    #############################################################################
    v = config['momentum']*config['velocity']-config['learning_rate']*dw
    next_w = w+v
    #############################################################################
    #                                                         结束编码                                                                            #
    #############################################################################
    # 更新速度
    config['velocity'] = v

    return next_w, config



def rmsprop(w, dw, config=None):
    """
    RMSProp更新规则

    config 使用格式:
    - learning_rate: 学习率.
    - decay_rate:历史累积梯度衰减率因子,取值为[0,1]
    - epsilon: 避免除零异常的小数.
    - cache:历史梯度缓存.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))

    next_w = None
    #############################################################################
    #                                                 任务：实现 RMSprop 更新                                                     #
    #    将更新后的权重保存在next_w中，将历史梯度累积存放在config['cache']中。        #
    #############################################################################
    config['cache'] = config['decay_rate']*config['cache']+(1-config['decay_rate'])*dw**2
    next_w = w-config['learning_rate']*dw/(np.sqrt(config['cache']+config['epsilon']))
    #############################################################################
    #                                                         结束编码                                                                            #
    #############################################################################
    return next_w, config


def adam(w, dw, config=None):
    """
    使用 Adam更新规则 ,融合了“热身”更新操作。
    
    config 使用格式:
    - learning_rate: 学习率.
    - beta1: 动量衰减因子.
    - beta2: 学习率衰减因子.
    - epsilon: 防除0小数.
    - m: 梯度.
    - v: 梯度平方.
    - t: 迭代次数.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)
    
    next_w = None
    #############################################################################
    #                                                    任务：实现Adam更新                                                             #                             
    #         将更新后的权重存放在next_w中，记得将m,v,t存放在相应的config中                 #
    #############################################################################
    config['t'] += 1
    beta1 = config['beta1']
    beta2 = config['beta2']
    epsilon = config['epsilon']
    learning_rate = config['learning_rate']
    config['m'] = beta1*config['m']+(1-beta1)*dw
    config['v'] = beta2*config['v']+(1-beta2)*dw**2
    mb = config['m']/(1-beta1**config['t'])
    vb = config['v']/(1-beta2**config['t'])
    next_w = w-learning_rate*mb/(np.sqrt(vb)+epsilon)
    #############################################################################
    #                                                        结束编码                                                                             #
    #############################################################################
    return next_w, config

