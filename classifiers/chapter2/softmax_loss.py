#-*- coding: utf-8 -*-
import numpy as np

def softmax_loss_naive(W, X, y, reg):
    """
    使用显式循环版本计算Softmax损失函数
    N表示：数据个数，D表示：数据维度，C：表示数据类别个数。
    Inputs:
    - W: 形状(D, C) numpy数组，表示分类器权重（参数）.
    - X: 形状(N, D) numpy数组，表示训练数据.
    - y: 形状(N,) numpy数组，表示数据类标。
        其中 y[i] = c 意味着X[i]为第c类数据，c取值为[0,c)
    - reg: 正则化惩罚系数
    Returns  二元组(tuple):
    - loss,数据损失值
    - dW,权重W所对应的梯度，其形状和W相同
    """
    # 初始化损失值与梯度.
    loss = 0.0  
    dW = np.zeros_like(W)
    #############################################################################        
    #  任务：使用显式循环实现softmax损失值loss及相应的梯度dW 。                 #
    #  温馨提示： 如果不慎,将很容易造成数值上溢。别忘了正则化哟。               #
    #############################################################################
    # =============================================================================
    # 第一种
    # =============================================================================
    # num_train = X.shape[0]  # 获取训练样本数
    # num_class = W.shape[1]  # 获取分类总数
    # for i in range(num_train):  
    #     s = X[i].dot(W)  # (1, C) 每类的可能性
    #     scores = s - max(s)  # 关注最高分
    #     #使用指数函数，在不影响单调性的情况下，使相对得分更明显
    #     scores_E = np.exp(scores)  
    #     # 计算总得分
    #     Z = np.sum(scores_E)
    #     # 找到目标值
    #     score_target = scores_E[y[i]]
    #     # 计算出损失值
    #     loss += -np.log(score_target/Z)
    #     # 计算梯度值
    #     for j in range(num_class):
    #         if j == y[i]:
    #             dW[:, j] += -(1-scores_E[j]/Z)*X[i]
    #         else:
    #             dW[:, j] += X[i]*scores_E[j]/Z
    # # 使用平均损失，再引入正则化
    # loss = loss/num_train+0.5*reg*np.sum(W*W)
    # # 使用平均梯度，再引入正则化
    # dW = dW/num_train+reg*W

    # =============================================================================
    # 第二种
    # =============================================================================
    N = X.shape[0]  # 获取训练样本数
    C = W.shape[1]  # 获取分类总数

    result = X.dot(W)
    result -= np.max(result,axis=1,keepdims=True)#避免指数太大，导致计算太大，内存不够

    for i in range(N):
        # 计算函数值
        soft_max = np.exp(result[i][y[i]])/np.sum(np.exp(result[i]))
        # 计算出损失值
        loss += -np.log(soft_max)

        for j in range(C):
            
            if j==y[i]:
                dW[:,j] += -X[i].T
            dW[:,j] += (X[i].T * np.exp(result[i][j])) / (np.sum(np.exp(result[i])))

    loss/= N
    loss += reg*np.sum(W*W)
    dW /= N
    dW += reg*2*W
    #############################################################################
    #                           结束编码                                        #
    #############################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax损失函数，使用矢量计算版本.
    输入，输出格式与softmax_loss_naive相同
    """
    # 初始化损失值与梯度
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # 任务: 不使用显式循环计算softmax的损失值loss及其梯度dW.                    #
    # 温馨提示： 如果不慎,将很容易造成数值上溢。别忘了正则化哟。                #
    #############################################################################
    # 训练样本数
    num_train = X.shape[0]
    # 计算函数值
    s = np.dot(X, W)
    # 关注最高分
    scores = s-np.max(s, axis=1, keepdims=True)
    # 使用指数函数来扩大相对比
    scores_E = np.exp(scores)
    # 求和得到总值
    Z = np.sum(scores_E, axis=1, keepdims=True)
    # 计算每一类的可能性
    prob = scores_E/Z
    # 真实标签标记
    y_trueClass = np.zeros_like(prob)
    y_trueClass[range(num_train), y] = 1.0
    # 计算损失值
    loss += -np.sum(y_trueClass*np.log(prob))/num_train+0.5*reg*np.sum(W*W)
    # 计算梯度
    dW += -np.dot(X.T, y_trueClass-prob)/num_train+reg*W
    #############################################################################
    #                          结束编码                                         #
    #############################################################################
    return loss, dW
