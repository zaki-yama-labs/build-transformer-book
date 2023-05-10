import tensorflow as tf

# 重みのパラメーター
a = tf.constant([[1.,2.,3.], [3.,4.,5.], [5.,6.,7.], [7.,8.,9.]])
a_v = tf.Variable(a, name='a')

# バイアスパラメーター
b = tf.constant([0.1, 0.2, 0.3])
b_v = tf.Variable(b, name='b')

# 入力値
x = tf.constant([[1.,2.,3.,4.]])

tgt = [[10.,20.,30.]]
# 損失関数
loss = lambda: ((tf.matmul(x,a_v)+b_v)-tgt)**2

# 確率的勾配降下法アルゴリズム
opt = tf.keras.optimizers.SGD()

# 1つ勾配を適用
opt.minimize(loss, var_list=[a_v, b_v])

# 更新されたパラメーター
print(a_v, b_v)
