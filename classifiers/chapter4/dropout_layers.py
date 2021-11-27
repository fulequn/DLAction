#-*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.realpath(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
from layers import *

def dropout_forward(x, dropout_param):
    """
    执行dropout前向传播过程。
    Inputs:
    - x: 输入数据
    - dropout_param: 字典类型的dropout参数，使用下列键值:
        - p: dropout激活参数，每个神经元的激活概率p。
        - mode: 'test'或'train'，train：使用激活概率p与神经元进行"and"运算;
                                                            test：去除激活概率p仅仅返回输入值。
        - seed: 随机数生成种子. 
    Outputs:
    - out: 和输入数据形状相同。
    - cache:元组(dropout_param, mask). 
                    训练模式：掩码mask用于激活该层神经元为“1”激活，为“0”抑制。
                    测试模式：去除掩码操作。
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    # 设置随机种子，控制输出稳定
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None  # 掩码层,用于筛选神经元
    out = None  # 输出层结果

    # 训练阶段
    if mode == 'train':
        ###########################################################################
        #                                任务：执行训练阶段dropout前向传播。                                            #
        ###########################################################################
        # 使用mask=mask/p，这样就可以保证输出均值与输入均值相同
        mask = (np.random.rand(*x.shape)<p)/p
        out = x*mask
        ###########################################################################
        #                                                     结束编码                                                                            #
        ###########################################################################
    # 测试阶段
    elif mode == 'test':
        ###########################################################################
        #                             任务： 执行测试阶段dropout前向传播。                                            #
        ###########################################################################
        out = x
        ###########################################################################
        #                                                     结束编码                                                                            #
        ###########################################################################
    # 输入参数与掩码层
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache


def dropout_backward(dout, cache):
    """
    dropout反向传播过程。
    Inputs:
    - dout: 上层梯度，形状和其输入相同。
    - cache: 前向传播中的缓存(dropout_param, mask)。
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']
    
    dx = None
    if mode == 'train':
        ###########################################################################
        #                                            任务：实现dropout反向传播                                                    #
        ###########################################################################
        dx = dout*mask
        ###########################################################################
        #                                                        结束编码                                                                         #
        ###########################################################################
    elif mode == 'test':
        dx = dout
    return dx

def affine_relu_dropout_forward(x,w,b,dropout_param):
    """
    组合affine_relu_dropout前向传播过程。
    Inputs:
    - x: 输入数据，其形状为(N, d_1, ..., d_k)的numpy数组。
    - w: 权重矩阵，其形状为(D,M)的numpy数组，
             D表示输入数据维度，M表示输出数据维度。
             可以将D看成输入的神经元个数，M看成输出神经元个数。
    - b: 偏置向量，其形状为(M,)的numpy数组。
    
    - dropout_param: 字典类型的dropout参数，使用下列键值:
        - p: dropout激活参数，每个神经元的激活概率p。
        - mode: 'test'或'train'，train：使用激活概率p与神经元进行"and"运算;
                                                            test：去除激活概率p仅仅返回输入值。
        - seed: 随机数生成种子.    

    Outputs:
    - out: 和输入数据形状相同。
    - cache:缓存包含(cache_affine,cache_relu,cache_dropout)
                    cache_affine：仿射前向传播的各项缓存；
                    cache_relu：ReLU前向传播的各项缓存；
                    cache_dropout：dropout前向传播的各项缓存。
    """ 
    out_dropout = None
    cache = None
    #############################################################################
    #                             任务: 实现 affine_relu_dropout 神经元前向传播.                            #
    #                 注意：你需要调用affine_forward以及relu_forward函数，                            #
    #                            并将各自的缓存保存在cache中                                                                    #
    #############################################################################    
    out_affine, cache_affine = affine_forward(x, w, b)
    out_relu, cache_relu = relu_forward(out_affine)
    out_dropout, cache_dropout = dropout_forward(out_relu, dropout_param)
    cache = (cache_affine, cache_relu, cache_dropout)
    ###########################################################################
    #                                                        结束编码                                                                         #
    ###########################################################################        
    return out_dropout,cache

def affine_relu_dropout_backward(dout,cache):
    """
     affine_relu_dropout神经元的反向传播过程。
     
    Input:
    - dout: 形状为(N, M)的上层梯度。
    - cache: 缓存(cache_affine,cache_relu,cache_dropout)。

    Returns:
    - dx: 输入数据x的梯度，其形状为(N, d1, ..., d_k)
    - dw: 权重矩阵w的梯度，其形状为(D,M)
    - db: 偏置项b的梯度，其形状为(M,)
    """    
    cache_affine,cache_relu,cache_dropout = cache
    dx,dw,db=None,None,None
    ###########################################################################
    #                             任务：实现affine_relu_dropout反向传播                                         #
    ###########################################################################    
    ddropout = dropout_backward(dout, cache_dropout)
    drelu = relu_backward(ddropout, cache_relu)
    dx, dw, db = affine_backward(drelu, cache_affine)     
    ###########################################################################
    #                                                    结束编码                                                                            #
    ###########################################################################
    return dx,dw,db
