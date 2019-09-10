import d2lzh as d2l
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import nn
import time


#  构建AlexNet网络
net = nn.Sequential()
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'))
net.add(nn.MaxPool2D(pool_size=3, strides=2))
net.add(nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'))
net.add(nn.MaxPool2D(pool_size=3, strides=2))
net.add(nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'))
net.add(nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'))
net.add(nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'))
net.add(nn.MaxPool2D(pool_size=3, strides=2))
net.add(nn.Dense(4096, activation='relu'))
net.add(nn.Dropout(0.5))
net.add(nn.Dense(4096, activation='relu'))
net.add(nn.Dropout(0.5))
net.add(nn.Dense(10))


# def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join(
#         '~', '.mxnet', 'datasets', 'fashion-mnist')):
#     root = os.path.expanduser(root)  # 展开用户路径'~'
#     transformer = []
#     if resize:
#         transformer += [gdata.vision.transforms.Resize(resize)]
#     transformer += [gdata.vision.transforms.ToTensor()]
#     transformer = gdata.vision.transforms.Compose(transformer)
#     mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
#     mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
#     num_workers = 0 if sys.platform.startswith('win32') else 4
#     train_iter = gdata.DataLoader(
#         mnist_train.transform_first(transformer), batch_size, shuffle=True,
#         num_workers=num_workers)
#     test_iter = gdata.DataLoader(
#         mnist_test.transform_first(transformer), batch_size, shuffle=False,
#         num_workers=num_workers)
#     return train_iter, test_iter


batch_size = 64
# 构建数据集，将原来28x28的图片放大到224x224
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)


# lr, num_epochs, ctx = 0.01, 5, d2l.try_gpu()
# net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
# d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
ctx = d2l.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
print('trying on', ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

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
