#-*- coding: utf-8 -*-
import numpy as np

"""

频繁使用的神经网络一阶梯度更新规则。每次更新接收：当前的网络权重，
训练获得的梯度，以及相关配置进行权重更新。
def update(w, dw, config = None):
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

  config :
  - learning_rate: 学习率.
  """
  if config is None: 
      config = {}
  config.setdefault('learning_rate', 1e-2)

  w -= config['learning_rate'] * dw
  return w, config

