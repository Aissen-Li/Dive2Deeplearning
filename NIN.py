import d2lzh as d2l
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import nn
import time


def nin_block(num_channels, kernel_size, padding, strides=1, max_pooling=True):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size,
                      strides, padding, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    if max_pooling:
        blk.add(nn.MaxPool2D(pool_size=3, strides=2))
    return blk


net = nn.Sequential()
net.add(nin_block(96, 11, 0, 4, True),
        nin_block(256, 5, 2, 1, True),
        nin_block(384, 3, 1, 1, True),
        nn.Dropout(0.5),
        nin_block(10, 3, 1, 1, False),
        nn.GlobalAvgPool2D(),
        nn.Flatten())

batch_size = 64
# 构建数据集，将原来28x28的图片放大到224x224
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

ctx = d2l.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
print('training on', ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

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