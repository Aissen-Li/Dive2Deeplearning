from mxnet import autograd, nd
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon


num_inputs = 2
num_examples = 1000
true_w = [2.1, -5]
true_b = 3.4
samples = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * samples[:, 0] + true_w[1] * samples[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
batch_size = 10
dataset = gdata.ArrayDataset(samples, labels)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize((init.Normal(sigma=0.01)))
loss = gloss.L2Loss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

num_epochs = 5
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(samples), labels)
    print('epoch {}, loss: {}'.format(epoch, l.mean().asnumpy()))

dense = net[0]
print('真实权重:{},训练权重:{}'.format(true_w, dense.weight.data()))
print('真是偏差:{},训练偏差:{}'.format(true_b, dense.bias.data()))