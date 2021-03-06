#-*- coding: utf-8 -*-
import numpy as np

def affine_forward(x, w, b):
    """
    计算神经网络当前层的前馈传播。该方法计算在全连接情况下的得分函数
    注：如果不理解affine仿射变换，简单的理解为在全连接情况下的得分函数即可

    输入数据x的形状为(N, d_1, ..., d_k)，其中N表示数据量，(d_1, ..., d_k)表示
    每一通道的数据维度。如果是图片数据就为(长，宽，色道)，数据的总维度就为
    D = d_1 * ... * d_k，因此我们需要数据整合成完整的（N,D)形式再进行仿射变换。
    
    Inputs:
    - x: 输入数据，其形状为(N, d_1, ..., d_k)的numpy array
    - w: 权重矩阵，其形状为(D,M)的numpy array，D表示输入数据维度，M表示输出数据维度
             可以将D看成输入的神经元个数，M看成输出神经元个数
    - b: 偏置向量，其形状为(M,)的numpy array
    
    Returns 元组:
    - out: 形状为(N, M)的输出结果
    - cache: 将输入进行缓存(x, w, b)
    """
    out = None
    # 任务: 实现全连接前向传播
    # 注：首先你需要将输入数据重塑成行。  
    N=x.shape[0]
    x_new=x.reshape(N,-1)#将x重塑成2维向量
    out=np.dot(x_new,w)+b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
 计算仿射层的反向传播.

    Inputs:
    - dout: 形状为(N, M)的上层梯度
    - cache: 元组:
        - x: (N, d_1, ... d_k)的输入数据
        - w: 形状为(D, M)的权重矩阵

    Returns 元组:
    - dx: 输入数据x的梯度，其形状为(N, d1, ..., d_k)
    - dw: 权重矩阵w的梯度，其形状为(D,M)
    - db: 偏置项b的梯度，其形状为(M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    # 注意：你需要将x重塑成(N,D)后才能计算各梯度，                                            #
    # 完梯度后你需要将dx的形状与x重塑成一样     
    db = np.sum(dout,axis=0)
    xx= x.reshape(x.shape[0],-1)
    dw = np.dot(xx.T,dout)
    dx = np.dot(dout,w.T)
    dx=np.reshape(dx,x.shape)
    return dx, dw, db


def relu_forward(x):
    """
    计算tified linear units (ReLUs)激活函数的前向传播，并保存相应缓存

    Input:
    - x: 输入数据

    Returns 元组:
    - out: 和输入数据x形状相同
    - cache: x
    """
    out = None
    # 实现ReLU 的前向传播.                                                                        #
    out =np.maximum(0,x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    计算 rectified linear units (ReLUs)激活函数的反向传播.

    Input:
    - dout: 上层误差梯度
    - cache: 输入 x,其形状应该和dout相同

    Returns:
    - dx: x的梯度
    """
    dx, x = None, cache
    # 实现 ReLU 反向传播.    
    dx=dout
    dx[x<=0]=0
    return dx

def affine_relu_forward(x, w, b):
    """
     ReLU神经元前向传播

    Inputs:
    - x: 输入到 affine层的数据
    - w, b:    affine层的权重矩阵和偏置向量

    Returns 元组:
    - out: Output from the ReLU的输出结果
    - cache: 前向传播的缓存
    """
    # 你需要调用affine_forward以及relu_forward函数，并将各自的缓存保存在cache中                                                                    #
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
     ReLU神经元的反向传播
     
    Input:
    - dout: 上层误差梯度
    - cache: affine缓存，以及relu缓存

    Returns:
    - dx: 输入数据x的梯度
    - dw: 权重矩阵w的梯度
    - db: 偏置向量b的梯度
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db




def softmax_loss(x, y):

    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx
