import mxnet
import d2lzh as d2l
from mxnet import autograd


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist((batch_size))

net = mxnet.gluon.nn.Sequential()  # 容器
net.add(mxnet.gluon.nn.Flatten())  # 平滑
net.add((mxnet.gluon.nn.Dense(10)))  # 全连接层,10个输出
net.initialize(mxnet.init.Normal(sigma=0.01))  # 初始化

loss = mxnet.gluon.loss.SoftmaxCrossEntropyLoss()  # 定义损失函数Softmax

trainer = mxnet.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})  # 训练器初始化

num_epochs = 5
lr = 0.1
for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y).sum()
        l.backward()  # 求导
        trainer.step(batch_size)  # 迭代并更新
        y = y.astype('float32')
        train_l_sum += l.asscalar()
        train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
        n += y.size
    test_acc_sum, test_n = 0.0, 0
    for test_X, test_y in test_iter:
        test_y = test_y.astype('float32')
        test_acc_sum += (net(test_X).argmax(axis=1) == test_y).sum().asscalar()
        test_n += test_y.size
    test_acc = test_acc_sum / test_n
    print('epoch {}, loss {:.4f}, train acc {:.3f}, test_acc {:.3f}'.format(epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))