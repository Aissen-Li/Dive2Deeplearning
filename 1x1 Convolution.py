import d2lzh as d2l
import mxnet as mx


def corr2d_multi_in(X, K):  # 多输入通道
    return mx.nd.add_n(*[d2l.corr2d(x, k) for x, k in zip(X, K)])


def corr2d_multi_in_out(X, K):  # 多输入多输出通道
    return mx.nd.stack(*[corr2d_multi_in(X, k) for k in K])


def corr2d_multi_in_out_1x1(X, K):  # 通过全连接层实现1x1卷积
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = mx.nd.dot(K, X)
    return Y.reshape((c_o, h, w))


X = mx.nd.random.uniform(shape=(3, 3, 3))
K = mx.nd.random.uniform(shape=(2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print('输入为{}'.format(X))
print('核为{}'.format(K))
print('通过全连接层算得{}'.format(Y1))
print('通过互相关运算得{}'.format(Y2))