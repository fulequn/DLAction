#-*- coding: utf-8 -*-
import numpy as np

from layers import *
from rnn_layers import *


class CaptioningRNN(object):
    """
    处理图片说明任务RNN网络
    注意：不使用正则化
    """
    
    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                             hidden_dim=128, cell_type='rnn'):
        """
        初始化CaptioningRNN 
        Inputs:
        - word_to_idx: 单词字典，用于查询单词索引对应的词向量
        - input_dim: 输入图片数据维度
        - wordvec_dim: 词向量维度.
        - hidden_dim: RNN隐藏层维度.
        - cell_type: 细胞类型; 'rnn' 或 'lstm'.
        """
        # 参数检验
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)
        
        # 初始化数据
        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}
        
        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)
        
        # 初始化词向量
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100
        
        # 初始化 CNN -> 隐藏层参数，用于将图片特征提取到RNN中
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # 初始化RNN参数
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)
        
        # 初始化输出层参数 
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)
            

    def loss(self, features, captions):
        """
        计算RNN或LSTM的损失值。
        Inputs:
        - features: 输入图片特征(N, D)。
        - captions: 图像文字说明(N, T)。 
            
        Returns 元组:
        - loss: 损失值。
        - grads:梯度。
        """
        #将文字切分为两段：captions_in除去最后一词用于RNN输入
        #captions_out除去第一个单词，用于RNN输出配对
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]
        
        # 掩码 
        mask = (captions_out != self._null)

        # 图像仿射转换矩阵
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        
        # 词嵌入矩阵
        W_embed = self.params['W_embed']

        # RNN参数
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # 隐藏层输出转化矩阵
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
        
        loss, grads = 0.0, {}
        ############################################################################
        #                        任务：实现CaptioningRNN传播                          #
        #         (1)使用仿射变换(features,W_proj,b_proj)，                           #
        #                     将图片特征输入进隐藏层初始状态h0(N,H)                      #
        #         (2)使用词嵌入层将captions_in中的单词索引转换为词向量(N,T,W)              #
        #         (3)使用RNN或LSTM处理词向量(N,T,H)                                    #
        #         (4)使用时序仿射传播temporal_affine_forward计算各单词得分(N,T,V)        #
        #         (5)使用temporal_softmax_loss计算损失值                              #
        ############################################################################
        # 1 使用仿射变换(features,W_proj,b_proj)，将图片特征输入进隐藏层初始状态h0(N,H)
        h0, cache_h0 = affine_forward(features, W_proj, b_proj)
        # 2 使用词嵌入层将captions_in中的单词索引转换为词向量(N,T,W)
        x, cache_embedding = word_embedding_forward(captions_in, W_embed)
        # 3 使用RNN或LSTM处理词向量(N,T,H)
        if self.cell_type == 'rnn':
            out_h, cache_rnn = rnn_forward(x, h0, Wx, Wh, b)
        elif self.cell_type == 'lstm':
            out_h, cache_rnn = lstm_forward(x, h0, Wx, Wh, b)
        else:
            raise ValueError('Invalid cell_type "%s"' % self.cell_type)
        # 4 使用时序仿射传播temporal_affine_forward计算各单词得分(N,T,V)
        yHat, cache_out = temporal_affine_forward(out_h, W_vocab, b_vocab)
        # 5 使用temporal_softmax_loss计算损失值
        loss, dy = temporal_softmax_loss(yHat, captions_out, mask, verbose=False)
        # 计算梯度
        dout_h, dW_vocab, db_vocab = temporal_affine_backward(dy, cache_out)
        # 输出层到隐藏层的反向传播
        if self.cell_type == 'rnn':
            dx, dh0, dWx, dWh, db = rnn_backward(dout_h, cache_rnn)
        elif self.cell_type == 'lstm':
            dx, dh0, dWx, dWh, db = lstm_backward(dout_h, cache_rnn)
        else:
            raise ValueError('Invalid cell_type "%s"' % self.cell_type)
        # 隐藏层到隐藏层自身的反向传播 
        dW_embed = word_embedding_backward(dx, cache_embedding)
        # 隐藏层到输入层的反向传播
        dfeatures, dW_proj, db_proj = affine_backward(dh0, cache_h0)
        # 记录梯度
        grads['W_proj'] = dW_proj
        grads['b_proj'] = db_proj
        grads['W_embed'] = dW_embed
        grads['Wx'] = dWx
        grads['Wh'] = dWh
        grads['b'] = db
        grads['W_vocab'] = dW_vocab
        grads['b_vocab'] = db_vocab
        ############################################################################
        #                          结束编码                                          #
        ############################################################################
        return loss, grads


    def sample(self, features, max_length=30):
        """
        测试阶段的前向传播过程，采样一批图片说明作为输入
        Inputs:
        - features: 图片特征(N, D).
        - max_length:生成说明文字的最大长度

        Returns:
        - captions: 说明文字的字典索引串(N, max_length)
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
        
        ###########################################################################
        #                             任务：测试阶段前向传播                                                                        #
        #    提示:(1)第一个单词应该是<START>标记，captions[:,0]=self._start                 #
        #             (2)当前单词输入为之前RNN的输出                                                                        #
        #        (3)前向传播过程为预测当前单词的下一个单词，                                                    #
        #         你需要计算所有单词得分，然后选取最大得分作为预测单词                                #
        #        (4)你无法使用rnn_forward 或 lstm_forward函数，                                                 #
        #        你需要循环调用rnn_step_forward或lstm_step_forward函数                                #
        ###########################################################################
        # 获取数据
        N, D = features.shape
        affine_out, affine_cache = affine_forward(features, W_proj, b_proj)
        prev_word_idx = [self._start]*N
        prev_h = affine_out
        prev_c = np.zeros(prev_h.shape)
        # 1第一个单词应该是<START>标记
        captions[:, 0] = self._start
        for i in range(1, max_length):
            # 2当前单词输入为之前RNN的输出 
            prev_word_embed = W_embed[prev_word_idx]
            # 4循环调用rnn_step_forward或lstm_step_forward函数
            if self.cell_type == 'rnn':
                next_h, rnn_step_cache = rnn_step_forward(prev_word_embed, prev_h,
                                                          Wx, Wh, b)
            elif self.cell_type == 'lstm':
                next_h, next_c, lstm_step_cache = lstm_step_forward(prev_word_embed, prev_h,
                                                          prev_c, Wx, Wh, b)
                prev_c = next_c
            else:
                raise ValueError('Invalid cell_type "%s"' % self.cell_type)
            vocab_affine_out, vocab_affine_out_cache = affine_forward(next_h, 
                            W_vocab, b_vocab)
            # 3计算所有单词得分，然后选取最大得分作为预测单词 
            captions[:, i] = list(np.argmax(vocab_affine_out, axis=1))
            prev_word_idx = captions[:, i]
            prev_h = next_h
        ############################################################################
        #                                                         结束编码                                                                         #
        ############################################################################
        return captions
