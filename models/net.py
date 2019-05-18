import tensorflow as tf
from .misc import softmax_cross_entropy, accuracy

class Net(object):
    def __init__(self):
        self.base = []
        self.bbd = []
        self.bbd_adv = []
        self.dbbd = []
        self.vib = []
        self.vib_adv = []
        self.sbp = []
        self.gend = []

    def params(self, mode=None, trainable=None):
        params = []
        for layer in getattr(self, mode):
            params += layer.params(trainable=trainable)
        return params

    def __call__(self, x, train=True, mode='base'):
        raise NotImplementedError()

    def apply(self, x, train, mode, l, mask_list=None):
        if mode == 'base':
            return x
        elif mode == 'bbd':
            if train:
                return self.bbd[l](x, train)
            else:
                return self.bbd[l](x, train)

                out_adv = self.bbd_adv[l](x, train)
                out = self.bbd[l](x, train)
                return tf.where(tf.equal(out_adv, tf.zeros_like(out_adv)), tf.zeros_like(out), out)
        elif mode == 'bbd_adv':
            if train:
                return self.bbd_adv[l](x, train)
            else:
                return self.bbd_adv[l](x, train)

                out_adv = self.bbd_adv[l](x, train)
                out = self.bbd[l](x, train)
                return tf.where(tf.equal(out, tf.zeros_like(out)), tf.zeros_like(out_adv), out_adv)
        elif mode == 'dbbd':
            z_in = self.bbd[l].mask(x, train)
            if mask_list is not None:
                mask_list.append(self.dbbd[l].mask(x, train))
            return self.dbbd[l](x, train, z_in)
        elif mode == 'vib':
            if train:
                return self.vib[l](x, train)
            else:
                return self.vib[l](x, train)

                out_adv = self.vib_adv[l](x, train)
                out = self.vib[l](x, train)
                return tf.where(tf.equal(out_adv, tf.zeros_like(out_adv)), tf.zeros_like(out), out)
        elif mode == 'vib_adv':
            if train:
                return self.vib_adv[l](x, train)
            else:
                return self.vib_adv[l](x, train)
                
                out_adv = self.vib_adv[l](x, train)
                out = self.vib[l](x, train)
                return tf.where(tf.equal(out, tf.zeros_like(out)), tf.zeros_like(out_adv), out_adv)
        elif mode == 'sbp':
            return self.sbp[l](x, train)
        elif mode == 'gend':
            return self.gend[l](x, train)
        else:
            raise ValueError('Invalid mode {}'.format(mode))

    def classify(self, x, y, train=True, mode='base'):
        x = self.__call__(x, train=train, mode=mode)
        cent = softmax_cross_entropy(x, y)
        acc = accuracy(x, y)
        return cent, acc

    def kl(self, mode=None):
        if mode is None:
            raise ValueError('Invalide mode {}'.format(mode))
        kl = [layer.kl() if hasattr(layer, 'kl') else 0. for layer in getattr(self, mode)]
        return tf.add_n(kl)

    def reg(self, y, train=True):
        key = 'train_probit' if train else 'test_probit'
        cent = [softmax_cross_entropy(getattr(layer, key), y) \
                for layer in self.dbbd]
        cent = tf.add_n(cent)/float(len(cent))
        return cent

    def n_active(self, mode=None):
        if mode is None:
            raise ValueError('Invalid mode {}'.format(mode))
        return [layer.n_active() for layer in getattr(self, mode)]

    def n_active_x(self):
        return [layer.n_active_x for layer in self.dbbd]

    def get_mask_diff(self, mode1, mode2):
        z1 = [layer.get_active() for layer in getattr(self, mode1)]
        z2 = [layer.get_active() for layer in getattr(self, mode2)]

        res = []
        for l in range(len(z1)):
            TT = tf.reduce_sum(tf.to_int32(tf.logical_and(z1[l], z2[l])))
            TF = tf.reduce_sum(tf.to_int32(tf.logical_and(z1[l], tf.logical_not(z2[l]))))
            FT = tf.reduce_sum(tf.to_int32(tf.logical_and(tf.logical_not(z1[l]), z2[l])))
            FF = tf.reduce_sum(tf.to_int32(tf.logical_and(tf.logical_not(z1[l]), tf.logical_not(z2[l]))))
            res.append(tf.stack([TT, TF, FT, FF]))
        return tf.stack(res)
