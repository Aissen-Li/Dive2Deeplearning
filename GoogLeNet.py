import d2lzh as d2l
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import nn
import time


class Inception(nn.Block):
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # path1, 1x1 Conv
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # path2, 1x1 Conv & 3x3 Conv
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,
                              activation='relu')
        # path3, 1x1 Conv & 5x5 Conv
        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,
                              activation='relu')
        # path4, 3x3 Max_pooling & 1x1 Conv
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return nd.concat(p1, p2, p3, p4, dim=1)  # 在通道维上连结输出,输出个数c1+c2[1]+c3[1]+c4


class GoogLeNet(nn.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(GoogLeNet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            # block1
            b1 = nn.Sequential()
            b1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
                   nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            # block2
            b2 = nn.Sequential()
            b2.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
                   nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
                   nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            # block3
            b3 = nn.Sequential()
            b3.add(Inception(64, (96, 128), (16, 32), 32),
                   Inception(128, (128, 192), (32, 96), 64),
                   nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            # block4
            b4 = nn.Sequential()
            b4.add(Inception(192, (96, 208), (16, 48), 64),
                   Inception(160, (112, 224), (24, 64), 64),
                   Inception(128, (128, 256), (24, 64), 64),
                   Inception(112, (144, 288), (32, 64), 64),
                   Inception(256, (160, 320), (32, 128), 128),
                   nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            # block5
            b5 = nn.Sequential()
            b5.add(Inception(256, (160, 320), (32, 128), 128),
                   Inception(384, (192, 384), (48, 128), 128),
                   nn.GlobalAvgPool2D())
            # block6
            b6 = nn.Sequential()
            b6.add(nn.Flatten(),
                   nn.Dense(num_classes))
            # chain block together
            self.net = nn.Sequential()
            self.net.add(b1, b2, b3, b4, b5, b6)

    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s' % (i+1, out.shape))
        return out


net = GoogLeNet(10)
batch_size = 128
# 构建数据集，将原来28x28的图片放大到96x96
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

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
