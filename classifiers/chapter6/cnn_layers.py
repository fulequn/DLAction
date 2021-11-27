#-*- coding: utf-8 -*-
import numpy as np
from layers import *
from bn_layers import *


def conv_forward_naive(x, w, b, conv_param):
    """
    卷积前向传播。
    Input:
    - x: 四维图片数据(N, C, H, W)分别表示(数量，色道，高，宽)
    - w: 四维卷积核(F, C, HH, WW)分别表示(下层色道，上层色道，高，宽)
    - b: 偏置项(F,)
    - conv_param: 字典型参数表，其键值为:
        - 'stride':跳跃数据卷积的跨幅数量
        - 'pad':输入数据的零填充数量

    Returns 元组型:
    - out: 输出数据(N, F, H', W') ，其中 H' 和 W' 分别为：
        H' = 1 + (H + 2 * pad - HH) / stride
        W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    #############################################################################
    #                         任务: 实现卷积层的前向传播                                                                    #
    #                    提示: 你可以使用np.pad函数进行零填充                                                         #
    #############################################################################
    # 获取数据的各种数据量
    # 数量，色道，高，宽
    N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    # 下层色道，高，宽
    F,HH,WW = w.shape[0],w.shape[2],w.shape[3]
    # 输入数据的零填充数量
    pad = conv_param['pad']
    # 跳跃数据进行卷积的跨幅数量
    stride = conv_param['stride']
    # 进行填充
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant')
    
    # 计算循环次数
    Hhat = int(1 + (H + 2 * pad - HH) / stride)
    What= int(1 + (W + 2 * pad - WW) / stride)
    # 输出值
    out = np.zeros([N,F,Hhat,What])
    # 遍历所有数据的下层色道的高和宽
    for n in range(N):
        for f in range(F):
            for i in range(Hhat):
                for j in range(What):
                    xx =x_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
                    out[n,f,i,j] =np.sum(xx*w[f])+b[f]
    #############################################################################
    #                                                         结束编码                                                                            #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_forward_fast(x, w, b, conv_param):
    '''
    卷积前向传播的快速版本

    Parameters
    ----------
    x : 四维图片数据(N, C, H, W)分别表示(数量，色道，高，宽)
    w : 四维卷积核(F, C, HH, WW)分别表示(下层色道，上层色道，高，宽)
    b : 偏置项(F,)
    conv_param : 字典型参数表，其键值为:
        - 'stride':跳跃数据卷积的跨幅数量
        - 'pad':输入数据的零填充数量

    Returns
    -------
    out : 输出数据(N, F, H', W') ，其中 H' 和 W' 分别为：
        H' = 1 + (H + 2 * pad - HH) / stride
        W' = 1 + (W + 2 * pad - WW) / stride
    cache : (x, w, b, conv_param)

    '''
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    assert (W + 2 * pad - WW) % stride == 0, '宽度异常'
    assert (H + 2 * pad - HH) % stride == 0, '高度异常'
    # 零填充
    p = pad
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), 
                                        mode='constant') 
    # 计算输出维度
    H += 2 * pad
    W += 2 * pad
    out_h = int((H - HH) / stride + 1)
    out_w = int((W - WW) / stride + 1)
    shape = (C, HH, WW, N, out_h, out_w)
    strides = (H * W, W, 1, C * H * W, stride * W, stride)
    strides = x.itemsize * np.array(strides)
    x_stride = np.lib.stride_tricks.as_strided(x_padded,
                                shape=shape, strides=strides)
    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = (C * HH * WW, N * out_h * out_w)
    # 将所有卷积核重塑成一行
    res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)
    # 重塑输出
    res.shape = (F, N, out_h, out_w)
    out = res.transpose(1, 0, 2, 3)
    out = np.ascontiguousarray(out)
    cache = (x, w, b, conv_param)
    return out, cache



def conv_backward_naive1(dout, cache):
    """
    卷积层反向传播显式循环版本

    Inputs:
    - dout:上层梯度.
    - cache: 前向传播时的缓存元组 (x, w, b, conv_param) 

    Returns 元组:
    - dx:    x梯度
    - dw:    w梯度
    - db:    b梯度
    """
    dx, dw, db = None, None, None
    #############################################################################
    #                        任务 ：实现卷积层反向传播                                                             #
    #############################################################################
    x, w, b, conv_param = cache
    P = conv_param['pad']
    x_pad = np.pad(x,((0,),(0,),(P,),(P,)),'constant')
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, Hh, Hw = dout.shape
    S = conv_param['stride']
    dw = np.zeros((F, C, HH, WW))
    for fprime in range(F):
        for cprime in range(C):
            for i in range(HH):
                for j in range(WW):
                    sub_xpad =x_pad[:,cprime,i:i+Hh*S:S,j:j+Hw*S:S]
                    dw[fprime,cprime,i,j] = np.sum(
                            dout[:,fprime,:,:]*sub_xpad)

    
    db = np.zeros((F))
    for fprime in range(F):
                db[fprime] = np.sum(dout[:,fprime,:,:])
    dx = np.zeros((N, C, H, W))
    
    for nprime in range(N):
        for i in range(H):
            for j in range(W):
                for f in range(F):
                    for k in range(Hh):
                        for l in range(Hw):
                            mask1 = np.zeros_like(w[f,:,:,:])
                            mask2 = np.zeros_like(w[f,:,:,:])
                            if (i+P-k*S)<HH and (i+P-k*S)>= 0:
                                    mask1[:,i+P-k*S,:] = 1.0
                            if (j+P-l* S) < WW and (j+P-l*S)>= 0:
                                    mask2[:,:,j+P-l*S] = 1.0
                            w_masked=np.sum(w[f,:,:,:]*mask1*mask2,axis=(1,2))
                            dx[nprime,:,i,j] +=dout[nprime,f,k,l]*w_masked
    #############################################################################
    #                            结束编码                                                                         #
    #############################################################################
    return dx, dw, db


def conv_backward_naive(dout, cache):
    """
    卷积层反向传播

    Inputs:
    - dout:上层梯度.
    - cache: 前向传播时的缓存元组 (x, w, b, conv_param) 

    Returns 元组:
    - dx:    x梯度
    - dw:    w梯度
    - db:    b梯度
    """
    dx, dw, db = None, None, None
    #############################################################################
    #                        任务 ：实现卷积层反向传播                                                             #
    #############################################################################
    x, w, b, conv_param = cache
    # 初始化参数
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    # 计算循环次数
    H_out = int(1+(H+2*pad-HH)/stride)
    W_out = int(1+(W+2*pad-WW)/stride)
    # 进行0填充
    x_pad = np.pad(x,((0,), (0,), (pad,), (pad,)), 
                   mode='constant', constant_values=0)
    # 计算梯度
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    # 进行求解
    db = np.sum(dout, axis=(0, 2, 3))
    x_pad = np.pad(x,((0,), (0,), (pad,), (pad,)), 
                   mode='constant', constant_values=0)
    for i in range(H_out):
        for j in range(W_out):
            x_pad_masked = x_pad[:, :, i*stride:i*stride+HH, 
                                 j*stride:j*stride+WW]
            # 计算dw
            for k in range(F):
                dw[k, :, :, :] += np.sum(x_pad_masked*(dout[:, k, i, j])[:,
                            None, None, None], axis=0)
            
            # 计算dx_pad
            for n in range(N):
                dx_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += \
                    np.sum((w[:, :, :, :]*(dout[n, :, i, j])[:, None, None, None]),
                           axis=0)
                    
    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    #############################################################################
    #                              结束编码                                                                            #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    最大池化前向传播

    Inputs:
    - x: 数据 (N, C, H, W)
    - pool_param: 键值:
        - 'pool_height': 池化高
        - 'pool_width': 池化宽
        - 'stride': 步幅

    Returns 元组型:
    - out: 输出数据
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    #                        任务: 实现最大池化操作的前向传播                                                #
    #############################################################################
    # 初始化参数
    N, C, H, W = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    
    # 计算循环次数
    H_out = int((H-HH)/stride+1) 
    W_out = int((W-WW)/stride+1)
    out = np.zeros((N, C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            # 先找到对应区域
            x_masked = x[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
            # 选择其中最大的值
            out[:, :, i, j] = np.max(x_masked, axis=(2,3))
    #############################################################################
    #                            结束编码                                                                            #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_forward_fast(x, pool_param):
    '''
    最大池化前向传播的快速版本

    Parameters
    ----------
    x : 四维图片数据(N, C, H, W)分别表示(数量，色道，高，宽)
    pool_param : 字典型参数表，其键值为:
        - 'pool_height': 池化高
        - 'pool_width': 池化宽
        - 'stride': 步幅

    Returns
    -------
    out : 输出数据
    cache : (x, x_reshaped, out)
    '''
    # 初始化参数
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    assert pool_height == pool_width == stride, 'Invalid pool params'
    assert H % pool_height == 0
    assert W % pool_height == 0
    
    x_reshaped = x.reshape(N, C, int(H / pool_height), pool_height,
                                                int(W / pool_width), pool_width)
    out = x_reshaped.max(axis=3).max(axis=4)

    cache = (x, x_reshaped, out)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    最大池化反向传播.

    Inputs:
    - dout: 上层梯度
    - cache: 缓存 (x, pool_param)
    Returns:
    - dx:    x梯度
    """
    dx = None
    #############################################################################
    #                        任务：实现最大池化反向传播                                                                         #
    #############################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = int((H-HH)/stride+1)
    W_out = int((W-WW)/stride+1)
    dx = np.zeros_like(x)
    for i in range(H_out):
        for j in range(W_out):
            x_masked = x[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
            max_x_masked = np.max(x_masked, axis=(2, 3))
            temp_binary_mask = (x_masked == (max_x_masked)[:, :, None, None])
            dx[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += \
                temp_binary_mask*(dout[:, :, i, j])[:, :, None, None]
    #############################################################################
    #                          结束编码                                                                            #
    #############################################################################
    return dx


def max_pool_backward_fast(dout, cache):
    x, x_reshaped, out = cache
    dx_reshaped = np.zeros_like(x_reshaped)
    out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
    mask = (x_reshaped == out_newaxis)
    dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
    dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
    dx_reshaped[mask] = dout_broadcast[mask]
    dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
    dx = dx_reshaped.reshape(x.shape)
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    空间批量归一化前向传播
    
    Inputs:
    - x: 数据 (N, C, H, W)
    - gamma: 缩放因子 (C,)
    - beta: 偏移因子 (C,)
    - bn_param: 参数字典:
        - mode: 'train' or 'test';
        - eps: 数值稳定常数
        - momentum: 运行平均值衰减因子
        - running_mean: 形状为(D,) 的运行均值
        - running_var ：形状为 (D,) 的运行方差
        
    Returns 元组:
    - out:输出 (N, C, H, W)
    - cache: 用于反向传播的缓存
    """
    out, cache = None, None
    #############################################################################
    #                             任务：实现空间BN算法前向传播                                                                #
    #            提示：你只需要重塑数据，调用 batchnorm_forward函数即可                             #
    #############################################################################
    N, C, H, W = x.shape
    temp_output, cache = batchnorm_forward(
        x.transpose(0, 3, 2, 1).reshape(N*H*W, C), gamma, beta, bn_param)
    out = temp_output.reshape(N, W, H, C).transpose(0, 3, 2, 1)
    #############################################################################
    #                          结束编码                                                                            #
    #############################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
        空间批量归一化反向传播
    
    Inputs:
    - dout: 上层梯度 (N, C, H, W)
    - cache: 前向传播缓存
    
    Returns 元组:
    - dx:输入梯度 (N, C, H, W)
    - dgamma: gamma梯度 (C,)
    - dbeta: beta梯度 (C,)
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    #                        任务：实现空间BN算法反向传播                                                                     #
    #                        提示：你只需要重塑数据调用batchnorm_backward_alt函数即可             #                            
    #############################################################################    
    N, C, H, W = dout.shape
    dx_temp, dgamma, dbeta = batchnorm_backward_alt(
        dout.transpose(0, 3 , 2, 1).reshape((N*H*W, C)), cache)
    dx = dx_temp.reshape(N, W, H, C).transpose(0, 3, 2, 1)
    #############################################################################
    #                       结束编码                                                                                     #
    #############################################################################
    return dx, dgamma, dbeta
    

def conv_relu_forward(x, w, b, conv_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_naive(da, conv_cache)
    return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    '''
    完整卷积层的反向传播

    Parameters
    ----------
    dout : 上层梯度 (N, C, H, W)
    cache : (conv_cache, relu_cache, pool_cache)

    Returns
    -------
    dx : x的梯度
    dw : w的梯度
    db : b的梯度
    '''
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_naive(da, conv_cache)
    return dx, dw, db
