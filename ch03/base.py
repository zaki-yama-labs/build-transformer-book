import tensorflow as tf


# 全結合層
class fc(tf.keras.layers.Layer):
    def __init__(self, n_inputs, n_feats, name, w_init_stdev=0.02):
        super(fc, self).__init__(name=name)

        nx = n_inputs
        nf = n_feats
        # 重み
        wb = tf.random.normal([nx, nf], stddev=w_init_stdev)
        self.w = tf.Variable(wb, name=f'{name}_w')
        # バイアス
        bb = tf.zeros([nf])
        self.b = tf.Variable(bb, name=f'{name}_b')

    def call(self, x):
        w = self.w
        b = self.b
        # パーセプトロンを実行
        c = tf.matmul(x, w) + b
        return c
