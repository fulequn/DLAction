#-*- coding: utf-8 -*-
import numpy as np

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    RNN单步前向传播，使用tanh激活单元
    Inputs:
    - x: 当前时间步数据输入(N, D).
    - prev_h: 前一时间步隐藏层状态 (N, H)
    - Wx: 输入层到隐藏层连接权重(D, H)
    - Wh:隐藏层到隐藏层连接权重(H, H)
    - b: 隐藏层偏置项(H,)

    Returns 元组:
    - next_h: 下一隐藏层状态(N, H)
    - cache: 缓存
    """
    next_h, cache = None, None
    ##############################################################################
    #                        任务：实现RNN单步前向传播                              #
    #                         将输出值储存在next_h中，                              #
    #                 将反向传播时所需的各项缓存存放在cache中                         #
    ##############################################################################
    # 计算神经元输入
    a = prev_h.dot(Wh)+x.dot(Wx)+b
    # 神经元激活
    next_h = np.tanh(a)
    # 保留过程中的数据
    cache = (x, prev_h, Wh, Wx, b, next_h)    
    ##############################################################################
    #                      结束编码                                               #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    RNN单步反向传播。
    Inputs:
    - dnext_h: 后一时间片段的梯度。
    - cache: 前向传播时的缓存。
    
    Returns 元组:
    - dx: 数据梯度(N, D)。
    - dprev_h: 前一时间片段梯度(N, H)。
    - dWx: 输入层到隐藏层权重梯度(D,H)。
    - dWh:    隐藏层到隐藏层权重梯度(H, H)。
    - db: 偏置项梯度(H,)。
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    #                            任务：实现RNN单步反向传播                           #
    #            提示：tanh(x)梯度:    1 - tanh(x)*tanh(x)                         # 
    ##############################################################################
    # 获取缓存数据
    x, prev_h, Wh, Wx, b, next_h = cache
    # 根据链式求导法则依次计算各个变量的梯度
    dscores = dnext_h*(1-next_h*next_h)
    dWx = np.dot(x.T, dscores)
    db = np.sum(dscores, axis=0)
    dWh = np.dot(prev_h.T, dscores)
    dx = np.dot(dscores, Wx.T)
    dprev_h = np.dot(dscores, Wh.T)
    ##############################################################################
    #                                                             结束编码                                                                         #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    RNN前向传播。
    Inputs:
    - x: 完整的时序数据 (N, T, D)。
    - h0: 隐藏层初始化状态 (N, H)。
    - Wx: 输入层到隐藏层权重 (D, H)。
    - Wh:    隐藏层到隐藏层权重(H, H)。
    - b: 偏置项(H,)。
    
    Returns 元组:
    - h: 所有时间步隐藏层状态(N, T, H)。
    - cache: 反向传播所需的缓存。
    """
    h, cache = None, None
    ##############################################################################
    #                              任务：实现RNN前向传播。                           #
    #                提示： 使用前面实现的rnn_step_forward 函数。                     #
    ##############################################################################
    # 获取数据维度
    N, T, D = x.shape
    (H, ) = b.shape
    # 初始化h
    h = np.zeros((N, T, H))
    # 获取默认隐藏层状态
    prev_h = h0
    # 遍历所有时间
    for t in range(T):
        # 获取当前时间片段
        xt = x[:, t, :]
        # 计算每一个片段
        next_h, _ = rnn_step_forward(xt, prev_h, Wx, Wh, b)
        # 更新状态
        prev_h = next_h
        # 保留结果
        h[:, t, :] = prev_h
    # 数据缓存，
    cache = (x, h0, Wh, Wx, b, h)
    ##############################################################################
    #                              结束编码                                                                                 #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    RNN反向传播。
    Inputs:
    - dh: 隐藏层所有时间步梯度(N, T, H)。
    Returns 元组:
    - dx: 输入数据时序梯度(N, T, D)。
    - dh0: 初始隐藏层梯度(N, H)。
    - dWx: 输入层到隐藏层权重梯度(D, H)。
    - dWh: 隐藏层到隐藏层权重梯度(H, H)。
    - db: 偏置项梯度(H,)。
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    #                              任务：实现RNN反向传播。                           #
    #                        提示：使用 rnn_step_backward函数。                     #
    ##############################################################################
    # 获取缓存数据
    x, h0, Wh, Wx, b, h = cache 
    # 获取数据维度
    N, T, H = dh.shape
    _, _, D = x.shape
    # 得到最后的细胞状态
    next_h = h[:, T-1, :]
    # 初始化
    dprev_h = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros((H,))
    # 遍历所有时间片段
    for t in range(T):
        # 当前处理的时间片段（从后往前）
        t = T-1-t
        # 获取对应的数据
        xt = x[:, t, :]
        # 最初时间片段的之前细胞状态默认为h0
        if t == 0:
            prev_h = h0
        else:
            prev_h = h[:, t-1, :]
        # 获取缓存数据 
        step_cache = (xt, prev_h, Wh, Wx, b, next_h)
        # 更新状态
        next_h = prev_h
        dnext_h = dh[:, t, :]+dprev_h
        # 进行反向传播
        dx[:, t, :], dprev_h, dWxt, dWht, dbt = rnn_step_backward(dnext_h, step_cache)
        # 状态累加
        dWx, dWh, db = dWx+dWxt, dWh+dWht, db+dbt
    # 记录h0的梯度
    dh0 = dprev_h
    ##############################################################################
    #                                    结束编码                                  #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    词嵌入前向传播，将数据矩阵中的N条长度为T的词索引转化为词向量。
    如：W[x[i,j]]表示第i条，第j时间步单词索引所对应的词向量。
    Inputs:
    - x: 整数型数组(N,T),N表示数据条数，T表示单条数据长度，
        数组的每一元素存放着单词索引，取值范围[0,V)。
    - W: 词向量矩阵(V,D)存放各单词对应的向量。
    
    Returns 元组:
    - out:输出词向量(N, T, D)。 
    - cache:反向传播时所需的缓存。
    """
    out, cache = None, None
    ##############################################################################
    #                           任务：实现词嵌入前向传播。                            #
    ##############################################################################
    # 获取数据维度
    N, T = x.shape
    V, D = W.shape
    # 初始化
    out = np.zeros((N, T, D))
    # 遍历所有数据
    for i in range(N):
        for j in range(T):
            # 将其转化为词向量
            out[i, j] = W[x[i, j]]
    cache = (x, W.shape)
    ##############################################################################
    #                                        结束编码                              #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    词嵌入反向传播
    
    Inputs:
    - dout: 上层梯度 (N, T, D)
    - cache:前向传播缓存
    
    Returns:
    - dW: 词嵌入矩阵梯度(V, D).
    """
    dW = None
    ##############################################################################
    #                          任务：实现词嵌入反向传播                               #
    #                     提示：你可以使用np.add.at函数                              #
    #            例如 np.add.at(a,[1,2],1)相当于a[1],a[2]分别加1                    #
    ##############################################################################
    x, W_shape = cache
    dW = np.zeros(W_shape)
    # np.add.at()是将传入的数组中制定下标位置的元素加上指定的值.
    np.add.at(dW, x, dout)
    ##############################################################################
    #                                                             结束编码                                                                         #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    数值稳定版本的sigmoid函数。
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    LSTM单步前向传播
    
    Inputs:
    - x: 输入数据 (N, D)
    - prev_h: 前一隐藏层状态 (N, H)
    - prev_c: 前一细胞状态(N, H)
    - Wx: 输入层到隐藏层权重(D, 4H)
    - Wh: 隐藏层到隐藏层权重 (H, 4H)
    - b: 偏置项(4H,)
    
    Returns 元组:
    - next_h:    下一隐藏层状态(N, H)
    - next_c:    下一细胞状态(N, H)
    - cache: 反向传播所需的缓存
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    #                            任务：实现LSTM单步前向传播。                        #
    #                 提示：稳定版本的sigmoid函数已经帮你实现，直接调用即可。            #
    #                             tanh函数使用np.tanh。                           #
    #############################################################################
    # 获取数据
    N, D = x.shape
    N, H = prev_h.shape
    # 计算输入门、遗忘门、输出门
    input_gate = sigmoid(np.dot(x, Wx[:, 0:H])+np.dot(prev_h, Wh[:, 0:H])+b[0:H])
    forget_gate = sigmoid(np.dot(x, Wx[:, H:2*H])+np.dot(prev_h, Wh[:, H:2*H])
                          +b[H:2*H])
    output_gate = sigmoid(np.dot(x, Wx[:, 2*H:3*H])+np.dot(prev_h, Wh[:, 2*H:3*H])
                          +b[2*H:3*H])
    # 计算输出单元
    input_data = np.tanh(np.dot(x, Wx[:, 3*H:4*H])+np.dot(prev_h, Wh[:, 3*H:4*H])
                         +b[3*H:4*H])
    # 更新细胞记忆
    next_c = forget_gate*prev_c+input_data*input_gate
    # 计算细胞输出
    next_scores_c = np.tanh(next_c)
    next_h = output_gate*next_scores_c
    cache = (x, Wx, Wh, b, input_data, input_gate, output_gate, forget_gate,
             prev_h, prev_c, next_scores_c)
    ##############################################################################
    #                             结束编码                                         #
    ##############################################################################
    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
     LSTM单步反向传播
    
    Inputs:
    - dnext_h: 下一隐藏层梯度 (N, H)
    - dnext_c: 下一细胞梯度 (N, H)
    - cache: 前向传播缓存
    
    Returns 元组:
    - dx: 输入数据梯度 (N, D)
    - dprev_h: 前一隐藏层梯度 (N, H)
    - dprev_c: 前一细胞梯度(N, H)
    - dWx: 输入层到隐藏层梯度(D, 4H)
    - dWh:    隐藏层到隐藏层梯度(H, 4H)
    - db:    偏置梯度(4H,)
    """
    dx, dprev_h, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    #                      任务：实现LSTM单步反向传播                               #
    #       提示：sigmoid(x)函数梯度：sigmoid(x)*(1-sigmoid(x))                    #
    #             tanh(x)函数梯度：     1-tanh(x)*tanh(x)                         #
    #############################################################################
    # 获取数据
    x, Wx, Wh, b, input_data, input_gate, output_gate, forget_gate, prev_h,\
        prev_c, next_scores_c = cache
    N, D = x.shape
    N, H = prev_h.shape
    # 初始化变量
    dWx = np.zeros((D, 4*H))
    dxx = np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    dhh = np.zeros((H, 4*H))
    db = np.zeros(4*H)
    dx = np.zeros((N, D))
    dprev_h = np.zeros((N, H))
    # 计算当前细胞的梯度
    dc_tem = dnext_c+dnext_h*(1-next_scores_c**2)*output_gate
    # 求解tanh层
    dprev_c = forget_gate*dc_tem
    dforget_gate = prev_c*dc_tem
    dinput_gate = input_data*dc_tem
    dinput = input_gate*dc_tem
    doutput_gate = next_scores_c*dnext_h
    # 求解sigmoid层
    dscores_in_gate = input_gate*(1-input_gate)*dinput_gate
    dscores_forget_gate = forget_gate*(1-forget_gate)*dforget_gate
    dscores_out_gate = output_gate*(1-output_gate)*doutput_gate
    dscores_in = (1-input_data**2)*dinput
    da = np.hstack((dscores_in_gate, dscores_forget_gate, dscores_out_gate, dscores_in))
    dWx = np.dot(x.T, da)
    dWh = np.dot(prev_h.T, da)
    db = np.sum(da, axis=0)
    dx = np.dot(da, Wx.T)
    dprev_h = np.dot(da, Wh.T)
    ##############################################################################
    #                           结束编码                                           #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    LSTM前向传播
    Inputs:
    - x: 输入数据 (N, T, D)
    - h0:初始化隐藏层状态(N, H)
    - Wx: 输入层到隐藏层权重 (D, 4H)
    - Wh: 隐藏层到隐藏层权重(H, 4H)
    - b: 偏置项(4H,)
    
    Returns 元组:
    - h: 隐藏层所有状态 (N, T, H)
    - cache: 用于反向传播的缓存
    """
    h, cache = None, None
    #############################################################################
    #                    任务： 实现完整的LSTM前向传播                              #
    #############################################################################
    # 获取数据
    N, T, D = x.shape
    H = int(b.shape[0]/4)
    # 初始化信息
    h = np.zeros((N, T, H))
    cache = {}
    prev_h = h0
    prev_c = np.zeros((N, H))
    # 遍历所有时序数据
    for t in range(T):
        # 当前数据
        xt = x[:, t, :]
        # 进行单步LSTM前向传播
        next_h, next_c, cache[t] = lstm_step_forward(xt, prev_h, prev_c, Wx, Wh, b)
        # 更新状态
        prev_h = next_h
        prev_c = next_c
        h[:, t, :] = prev_h
    ##############################################################################
    #                          结束编码                                            #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    LSTM反向传播
    Inputs:
    - dh: 各隐藏层梯度(N, T, H)
    - cache: V前向传播缓存
    
    Returns 元组:
    - dx: 输入数据梯度 (N, T, D)
    - dh0:初始隐藏层梯度(N, H)
    - dWx: 输入层到隐藏层权重梯度 (D, 4H)
    - dWh: 隐藏层到隐藏层权重梯度 (H, 4H)
    - db: 偏置项梯度 (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    #               任务：实现完整的LSTM反向传播                                     #
    #############################################################################
    # 获取数据
    N, T, H = dh.shape
    # 从最后一条开始更新
    x, Wx, Wh, b, input_data, input_gate, output_gate, forget_gate, prev_h, prev_c,\
        next_scores_c = cache[T-1]
    D = x.shape[1]
    # 初始化
    dprev_h = np.zeros((N, H))
    dprev_c = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    db = np.zeros((4*H,))
    # 遍历所有数据
    for t in range(T):
        # 选择当前时间（从后向前）
        t = T-1-t
        # 获取数据
        step_cache = cache[t]
        dnext_h = dh[:, t, :]+dprev_h
        dnext_c = dprev_c
        # 进行单步反向传播计算
        dx[:, t, :], dprev_h, dprev_c, dWxt, dWht, dbt = lstm_step_backward(dnext_h,
                        dnext_c, step_cache)
        # 更新参数
        dWx, dWh, db = dWx+dWxt, dWh+dWht, db+dbt
    # 更新h0梯度
    dh0 = dprev_h
    ##############################################################################
    #                            结束编码                                          #
    ##############################################################################
    
    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    时序隐藏层仿射传播：将隐藏层时序数据(N,T,D)重塑为(N*T,D)，
    完成前向传播后，再重塑回原型输出。

    Inputs:
    - x: 时序数据(N, T, D)。
    - w: 权重(D, M)。
    - b: 偏置(M,)。
    
    Returns 元组:
    - out: 输出(N, T, M)。
    - cache: 反向传播缓存。
    """
    N, T, D = x.shape
    M = b.shape[0]
    # Affine层
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    时序隐藏层仿射反向传播。

    Input:
    - dout:上层梯度 (N, T, M)。
    - cache: 前向传播缓存。

    Returns 元组:
    - dx: 输入梯度(N, T, D)。
    - dw: 权重梯度 (D, M)。
    - db: 偏置项梯度 (M,)。
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]
    # Affine层反向传播
    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    时序版本的Softmax损失和原版本类似，只需将数据(N, T, V)重塑为(N*T,V)即可。
    需要注意的是，对于NULL标记不计入损失值，因此，你需要加入掩码进行过滤。
    Inputs:
    - x: 输入数据得分(N, T, V)。
    - y: 目标索引(N, T)，其中0<= y[i, t] < V。
    - mask: 过滤NULL标记的掩码。
    Returns 元组:
    - loss: 损失值。
    - dx: x梯度。
    """
    # 获取必备信息
    N, T, V = x.shape
    
    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)
    
    # 和原有softmax类似，不足的部分使用NULL补充，计算的时候过滤
    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]
    
    # 是否打印
    if verbose: 
        print('dx_flat: ', dx_flat.shape)
    
    dx = dx_flat.reshape(N, T, V)
    
    return loss, dx

