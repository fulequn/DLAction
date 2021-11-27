#-*- coding: utf-8 -*-
import numpy as np

import updater


class Trainer(object):
    """
    使用形式:
    
    data = {
        'X_train': # 训练数据
        'y_train': # 训练类标
        'X_val': # 验证数据
        'X_train': # 验证类标
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    Trainer = Trainer(model, data,
                                    update_rule='sgd',
                                    updater_config={
                                        'learning_rate': 1e-3,
                                    },
                                    lr_decay=0.95,
                                    num_epochs=10, batch_size=100,
                                    print_every=100)
    Trainer.train()
    """

    def __init__(self, model, data, **kwargs):
        """
        构造一个新的Trainer实例
        必须参数:
        - model: 网络模型
        - data: 数据字典，其中:
            'X_train':    形状为(N_train, d_1, ..., d_k)训练数据
            'X_val':    形状为(N_val, d_1, ..., d_k) 验证数据
            'y_train':    形状为(N_train,) 训练数据类标
            'y_val':    形状为(N_val,) 验证数据类标
            
        可选参数:
        - update_rule: 更新规则，其存放在updater.py文件中，默认选项为'sgd'。
        - updater_config: 字典类型的，更新规则所对应的超参数配置，同见updater.py文件。
        - lr_decay: 学习率衰减系数。
        - batch_size: 批量数据大小
        - num_epochs: 训练周期
        - print_every: 整数型; 每迭代多少次进行打印一次中间结果
        - verbose: 布尔型; 是否在训练期间打印中间结果
        """
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        
        # 弹出可选参数，进行相关配置
        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.updater_config = kwargs.pop('updater_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        # 若可选参数错误，抛出异常
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)


        #确认updater中含有更新规则
        if not hasattr(updater, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(updater, self.update_rule)

        # 初始化相关变量
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # 对updater_config中的参数进行深拷贝
        self.updater_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.updater_config.items()}
            self.updater_configs[p] = d


    def _step(self):
        """
        执行单步梯度更新
        """
        # 采样批量数据
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # 计算损失及梯度
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # 更新参数
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.updater_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.updater_configs[p] = next_config


    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
     根据提供的数据检验精度，若数据集过大，可进行采样测试。
        
        Inputs:
        - X: 形状为(N, d_1, ..., d_k)的数据
        - y: 形状为 (N,)的数据类标
        - num_samples: 采样次数
        - batch_size:批量数据大小
            
        Returns:
        - acc: 测试数据正确率
        """
        
        # 对数据进行采样
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # 计算精度
        num_batches = int(N / batch_size)
        if N % batch_size != 0:
          num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc


    def train(self):
        """
        根据配置训练模型
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = int(self.num_epochs * iterations_per_epoch)
    
        for t in range(num_iterations):
            self._step()
    
            # 打印损失值
            if self.verbose and t % self.print_every == 0:
                print('(迭代 %d / %d) 损失值: %f' % (
                             t + 1, num_iterations, self.loss_history[-1]))

            # 更新学习率
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.updater_configs:
                    self.updater_configs[k]['learning_rate'] *= self.lr_decay


            #在训练的开始，末尾，每一轮训练周期检验精度
            first_it = (t == 0)
            last_it = (t == num_iterations + 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train,
                                                                                num_samples=1000)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print('(周期 %d / %d) 训练精度: %f; 验证精度: %f' % (
                                 self.epoch, self.num_epochs, train_acc, val_acc))

                # 记录最佳模型
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()
    
            # 训练结束后返回最佳模型
            self.model.params = self.best_params

