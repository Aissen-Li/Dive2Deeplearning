import d2lzh as d2l
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import nn
import time


class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1conv=False, **kwargs):  # 若输出通道不同strides等于2
        super(Residual, self).__init__(**kwargs)
        strides = 2 if use_1x1conv else 1  # 通过是否改变输出通道形状决定strides
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:  # 改变输出通道形状则使用1x1 Conv进行变换
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:  # 如果需要改变输出通道
            X = self.conv3(X)
        return nd.relu(Y + X)


net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))


def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True))
        else:
            blk.add(Residual(num_channels))
    return blk


net.add(resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2))
net.add(nn.GlobalAvgPool2D(), nn.Dense(10))

batch_size = 64
# 构建数据集，将原来28x28的图片放大到96x96
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

ctx = d2l.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
print('training on', ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05})

for epoch in range(3):
    train_loss_sum = 0
    train_acc_sum = 0
    n = 0
    start = time.time()
    for X, y in train_iter:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        with autograd.record():
            y_hat = net(X)
            loss = softmax_cross_entropy(y_hat, y).sum()
        loss.backward()
        trainer.step(batch_size)
        y = y.astype('float32')
        train_loss_sum += loss.asscalar()
        train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
        n += y.size
    test_acc = d2l.evaluate_accuracy(test_iter, net, ctx)
    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
          % (epoch + 1, train_loss_sum / n, train_acc_sum / n, test_acc, time.time() - start))