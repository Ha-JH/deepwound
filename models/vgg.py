from __future__ import print_function
from .net import Net
from .layers import *
from .bbdropout import BBDropout
from .dbbdropout import DBBDropout
from .vib import VIB
from .sbp import SBP
from .gendropout import GenDropout
from .misc import *

class VGG(Net):
    def __init__(self, n_classes, mode, mask=None,
            name='vgg', reuse=None):
        super(VGG, self).__init__()
        if mode==16:
            n_units = [64, 64, 128, 128, 256, 256, 256,
                    512, 512, 512, 512, 512, 512, 512, 512]
        elif mode==11:
            n_units = [64, 128, 256, 256,
                    512, 512, 512, 512, 512, 512]
        elif mode==13:
            n_units = [64, 64, 128, 128, 256, 256,
                    512, 512, 512, 512, 512, 512]
        elif mode==19:
            n_units = [64, 64, 128, 128, 256, 256, 256, 256,
                    512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
        else:
            raise ValueError('Invalid mode {}'.format(mode))
        self.mask = mask
        self.n_classes = n_classes

        def create_block(l, n_in, n_out):
            self.base.append(Conv(n_in, n_out, 3,
                name='conv'+str(l), padding='SAME'))
            self.base.append(BatchNorm(n_out, name='bn'+str(l)))
            self.bbd.append(BBDropout(n_out, name='bbd'+str(l),
                a_uc_init=2.0))
            self.bbd_adv.append(BBDropout(n_out, name='bbd_adv'+str(l),
                a_uc_init=2.0))
            self.dbbd.append(DBBDropout(n_out, name='dbbd'+str(l)))
            self.vib.append(VIB(n_out, name='vib'+str(l)))
            self.vib_adv.append(VIB(n_out, name='vib_adv'+str(l)))
            self.sbp.append(SBP(n_out, name='sbp'+str(l)))
            self.gend.append(GenDropout(n_out, name='gend'+str(l)))

        with tf.variable_scope(name, reuse=reuse):
            create_block(1, 3, n_units[0])
            for i in range(1, 13):
                create_block(i+1, n_units[i-1], n_units[i])

            self.bbd.append(BBDropout(n_units[13], name='bbd14'))
            self.bbd_adv.append(BBDropout(n_units[13], name='bbd_adv14'))
            self.dbbd.append(DBBDropout(n_units[13], name='dbbd14'))
            self.vib.append(VIB(n_units[13], name='vib14'))
            self.vib_adv.append(VIB(n_units[13], name='vib_adv14'))
            self.sbp.append(SBP(n_units[13], name='sbp14'))
            self.gend.append(GenDropout(n_units[13], name='gend14'))

            self.base.append(Dense(n_units[13], n_units[14], name='dense14'))
            self.base.append(BatchNorm(n_units[14], name='bn14'))

            self.bbd.append(BBDropout(n_units[14], name='bbd15'))
            self.bbd_adv.append(BBDropout(n_units[14], name='bbd_adv15'))
            self.dbbd.append(DBBDropout(n_units[14], name='dbbd15'))
            self.vib.append(VIB(n_units[14], name='vib15'))
            self.vib_adv.append(VIB(n_units[14], name='vib_adv15'))
            self.sbp.append(SBP(n_units[14], name='sbp15'))
            self.gend.append(GenDropout(n_units[14], name='gen15'))

            self.base.append(Dense(n_units[14], n_classes, name='dense15'))

    def __call__(self, x, train, mode='base', mask_list=[]):
        def apply_block(x, train, l, mode, p=None):
            conv = self.base[2*l-2]
            bn = self.base[2*l-1]
            x = self.apply(conv(x), train, mode, l-1, mask_list=mask_list)
            if mode == 'sbp':
                x = relu(bn(x, False))
            else:
                x = relu(bn(x, train))
            x = pool(x) if p is None else tf.layers.dropout(x, p, training=train)
            return x

        p_list = [0.3, None,
                0.4, None,
                0.4, 0.4, None,
                0.4, 0.4, None,
                0.4, 0.4, None]
        for l, p in enumerate(p_list):
            x = apply_block(x, train, l+1, mode, p=p)

        x = flatten(x)
        x = x if self.mask is None else tf.gather(x, self.mask, axis=1)
        x = tf.layers.dropout(x, 0.5, training=train) if mode=='base' else x
        x = self.base[2*13](self.apply(x, train, mode, 13, mask_list=mask_list))
        x = relu(self.base[2*13+1](x, False) if mode == 'sbp' else self.base[2*13+1](x, train))

        x = tf.layers.dropout(x, 0.5, training=train) if mode=='base' else x
        x = self.base[-1](self.apply(x, train, mode, 14, mask_list=mask_list))
        return x


class VGGBayes(Net):
    def __init__(self, n_classes, mode, sigma_prior=0.1, init_rho=-5.0, mask=None,
            name='vgg', reuse=None):
        super(VGGBayes, self).__init__()
        if mode==16:
            n_units = [64, 64, 128, 128, 256, 256, 256,
                    512, 512, 512, 512, 512, 512, 512, 512]
        elif mode==11:
            n_units = [64, 128, 256, 256,
                    512, 512, 512, 512, 512, 512]
        elif mode==13:
            n_units = [64, 64, 128, 128, 256, 256,
                    512, 512, 512, 512, 512, 512]
        elif mode==19:
            n_units = [64, 64, 128, 128, 256, 256, 256, 256,
                    512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
        else:
            raise ValueError('Invalid mode {}'.format(mode))

        self.mask = mask
        self.n_classes = n_classes

        def create_block(l, n_in, n_out):
            self.base.append(ConvBayes(sigma_prior, n_in, n_out, 3, init_rho=init_rho,
                name='conv'+str(l), padding='SAME'))
            self.base.append(BatchNormBayes(sigma_prior, n_out, name='bn'+str(l)))
            self.bbd.append(BBDropout(n_out, name='bbd'+str(l),
                a_uc_init=2.0))
            self.dbbd.append(DBBDropout(n_out, name='dbbd'+str(l)))
            self.vib.append(VIB(n_out, name='vib'+str(l)))
            self.sbp.append(SBP(n_out, name='sbp'+str(l)))
            self.gend.append(GenDropout(n_out, name='gend'+str(l)))

        with tf.variable_scope(name, reuse=reuse):
            create_block(1, 3, n_units[0])
            for i in range(1, len(n_units)-2):
                create_block(i+1, n_units[i-1], n_units[i])

            self.bbd.append(BBDropout(n_units[-2], name='bbd14'))
            self.dbbd.append(DBBDropout(n_units[-2], name='dbbd14'))
            self.vib.append(VIB(n_units[-2], name='vib14'))
            self.sbp.append(SBP(n_units[-2], name='sbp14'))
            self.gend.append(GenDropout(n_units[-2], name='gend14'))

            self.base.append(DenseBayes(sigma_prior, n_units[-2], n_units[-1], init_rho=init_rho, name='dense14'))
            self.base.append(BatchNormBayes(sigma_prior, n_units[-1], init_rho=init_rho, name='bn14'))

            self.bbd.append(BBDropout(n_units[-1], name='bbd15'))
            self.dbbd.append(DBBDropout(n_units[-1], name='dbbd15'))
            self.vib.append(VIB(n_units[-1], name='vib15'))
            self.sbp.append(SBP(n_units[-1], name='sbp15'))
            self.gend.append(GenDropout(n_units[-1], name='gen15'))

            self.base.append(DenseBayes(sigma_prior, n_units[-1], n_classes, init_rho=init_rho, name='dense15'))

    def __call__(self, x, train, mode='base', mask_list=[]):
        def apply_block(x, train, l, mode, p=None):
            conv = self.base[2 * l - 2]
            bn = self.base[2 * l - 1]
            x = self.apply(conv(x, mode=='base' and train), train, mode, l - 1, mask_list=mask_list)
            if mode == 'sbp':
                x = relu(bn(x, False))
            else:
                x = relu(bn(x, train))
            x = pool(x) if p is None else tf.layers.dropout(x, p, training=train)
            return x

        p_list = [0.3, None,
                0.4, None,
                0.4, 0.4, None,
                0.4, 0.4, None,
                0.4, 0.4, None]
        for l, p in enumerate(p_list):
            x = apply_block(x, train, l+1, mode, p=p)

        x = flatten(x)
        x = x if self.mask is None else tf.gather(x, self.mask, axis=1)
        x = tf.layers.dropout(x, 0.5, training=train) if mode=='base' else x
        x = self.base[2 * 13](self.apply(x, train, mode, 13, mask_list=mask_list), mode=='base' and train)
        x = relu(self.base[2 * 13 + 1](x, False) if mode == 'sbp' else self.base[2 * 13 + 1](x, train))

        x = tf.layers.dropout(x, 0.5, training=train) if mode=='base' else x
        x = self.base[-1](self.apply(x, train, mode, 14, mask_list=mask_list), mode=='base' and train)
        return x
