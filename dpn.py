from collections import OrderedDict

import chainer
import chainer.links as L
import chainer.functions as F

__all__ = ['DPN', 'dpn92', 'dpn98', 'dpn131', 'dpn107', 'dpns']


def dpn92(num_classes=1000):
    return DPN(num_init_features=64, k_R=96, G=32, k_sec=(3,4,20,3), inc_sec=(16,32,24,128), num_classes=num_classes)


def dpn98(num_classes=1000):
    return DPN(num_init_features=96, k_R=160, G=40, k_sec=(3,6,20,3), inc_sec=(16,32,32,128), num_classes=num_classes)


def dpn131(num_classes=1000):
    return DPN(num_init_features=128, k_R=160, G=40, k_sec=(4,8,28,3), inc_sec=(16,32,32,128), num_classes=num_classes)


def dpn107(num_classes=1000):
    return DPN(num_init_features=128, k_R=200, G=50, k_sec=(4,8,20,3), inc_sec=(20,64,64,128), num_classes=num_classes)


dpns = {
    'dpn92': dpn92,
    'dpn98': dpn98,
    'dpn107': dpn107,
    'dpn131': dpn131,
}


class Sequential(chainer.Chain):
    """A part of the code has been borrowed from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/container.py and https://github.com/musyoku/chainer-sequential-chain.
    """
    def __init__(self, *args):
        super(Sequential, self).__init__()
        assert len(args) > 0
        assert not hasattr(self, "layers")
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            self.layers = args[0].values()
            with self.init_scope():
                for key, layer in args[0].items():
                    if isinstance(layer, (chainer.Link, chainer.Chain, chainer.ChainList)):
                        setattr(self, key, layer)
        else:
            self.layers = args
            with self.init_scope():
                for idx, layer in enumerate(args):
                    if isinstance(layer, (chainer.Link, chainer.Chain, chainer.ChainList)):
                        setattr(self, str(idx), layer)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MaxPooling2D(object):
    def __init__(self, ksize, stride=None, pad=0, cover_all=True):
        self.args = [ksize, stride, pad, cover_all]

    def __call__(self, x):
        return F.max_pooling_2d(x, *self.args)


class GroupedConvolution2D(chainer.ChainList):
    def __init__(self, in_chs, out_chs, ksize=None, stride=1, pad=0, groups=1, nobias=False, initialW=None, initial_bias=None):
        assert in_chs % groups == 0
        assert out_chs % groups == 0
        group_in_chs = int(in_chs / groups)
        group_out_chs = int(in_chs / groups)
        super(GroupedConvolution2D, self).__init__(
            *[L.Convolution2D(group_in_chs, group_out_chs, ksize, stride, pad, nobias, initialW, initial_bias) for _ in range(groups)]
        )
        self.group_in_chs = group_in_chs

    def __call__(self, x):
        return F.concat([f(x[:,i*self.group_in_chs:(i+1)*self.group_in_chs,:,:]) for i, f in enumerate(self.children())], axis=1)


class DualPathBlock(chainer.Chain):
    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, G, _type='normal'):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c

        if _type is 'proj':
            key_stride = 1
            self.has_proj = True
        if _type is 'down':
            key_stride = 2
            self.has_proj = True
        if _type is 'normal':
            key_stride = 1
            self.has_proj = False

        with self.init_scope():
            if self.has_proj:
                self.c1x1_w = self.BN_ReLU_Conv(in_chs=in_chs, out_chs=num_1x1_c+2*inc, kernel_size=1, stride=key_stride)

            self.layers = Sequential(OrderedDict([
                ('c1x1_a', self.BN_ReLU_Conv(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)),
                ('c3x3_b', self.BN_ReLU_Conv(in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3, stride=key_stride, padding=1, groups=G)),
                ('c1x1_c', self.BN_ReLU_Conv(in_chs=num_3x3_b, out_chs=num_1x1_c+inc, kernel_size=1, stride=1)),
            ]))

    def BN_ReLU_Conv(self, in_chs, out_chs, kernel_size, stride, padding=0, groups=1):
        if groups==1:
            return Sequential(OrderedDict([
                ('norm', L.BatchNormalization(in_chs)),
                ('relu', F.relu),
                ('conv', L.Convolution2D(in_chs, out_chs, kernel_size, stride, padding, nobias=True)),
            ]))
        else:
            return Sequential(OrderedDict([
                ('norm', L.BatchNormalization(in_chs)),
                ('relu', F.relu),
                ('conv', GroupedConvolution2D(in_chs, out_chs, kernel_size, stride, padding, groups, nobias=True)),
            ]))
            

    def __call__(self, x):
        data_in = F.concat(x, axis=1) if isinstance(x, list) else x
        if self.has_proj:
            data_o = self.c1x1_w(data_in)
            data_o1 = data_o[:,:self.num_1x1_c,:,:]
            data_o2 = data_o[:,self.num_1x1_c:,:,:]
        else:
            data_o1 = x[0]
            data_o2 = x[1]

        out = self.layers(data_in)

        summ = data_o1 + out[:,:self.num_1x1_c,:,:]
        dense = F.concat([data_o2, out[:,self.num_1x1_c:,:,:]], axis=1)
        return [summ, dense]


class DPN(chainer.Chain):

    def __init__(self, num_init_features=64, k_R=96, G=32,
                 k_sec=(3, 4, 20, 3), inc_sec=(16,32,24,128), num_classes=1000):

        super(DPN, self).__init__()
        blocks = OrderedDict()

        # conv1
        blocks['conv1'] = Sequential(
            L.Convolution2D(3, num_init_features, ksize=7, stride=2, pad=3, nobias=True),
            L.BatchNormalization(num_init_features),
            F.relu,
            MaxPooling2D(ksize=3, stride=2, pad=1),
        )

        # conv2
        bw = 256
        inc = inc_sec[0]
        R = int((k_R*bw)/256)
        blocks['conv2_1'] = DualPathBlock(num_init_features, R, R, bw, inc, G, 'proj')
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0]+1):
            blocks['conv2_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal')
            in_chs += inc

        # conv3
        bw = 512
        inc = inc_sec[1]
        R = int((k_R*bw)/256)
        blocks['conv3_1'] = DualPathBlock(in_chs, R, R, bw, inc, G, 'down')
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1]+1):
            blocks['conv3_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal')
            in_chs += inc

        # conv4
        bw = 1024
        inc = inc_sec[2]
        R = int((k_R*bw)/256)
        blocks['conv4_1'] = DualPathBlock(in_chs, R, R, bw, inc, G, 'down')
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2]+1):
            blocks['conv4_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal')
            in_chs += inc

        # conv5
        bw = 2048
        inc = inc_sec[3]
        R = int((k_R*bw)/256)
        blocks['conv5_1'] = DualPathBlock(in_chs, R, R, bw, inc, G, 'down')
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3]+1):
            blocks['conv5_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal')
            in_chs += inc

        with self.init_scope():
            self.features = Sequential(blocks)
            self.classifier = L.Linear(in_chs, num_classes)


    def __call__(self, x, t):
        features = F.concat(self.features(x), axis=1)
        out = F.average_pooling_2d(features, ksize=7)
        out = self.classifier(out)

        loss = F.softmax_cross_entropy(out, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(out, t)}, self)
        return loss
