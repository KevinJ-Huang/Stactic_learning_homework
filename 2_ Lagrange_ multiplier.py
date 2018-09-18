#### method1
import tensorflow as tf

x1 = tf.Variable(tf.random_normal(shape=[1],mean=-0.1,stddev=0.01))
x2 = -x1
loss = 10 - x1*x1 - x2*x2
train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        xs,loss_show = sess.run([x1,loss])
        if xs*xs+xs <= 0 :
            _= sess.run(train_op)
        print('step =%d,loss=%.8f,x1=%.8f'%(i,loss_show,xs))


####method2
from sympy import *

x1 = Symbol("x1")
x2 = Symbol("x2")
a = Symbol("a")
b = Symbol("b")
f = 10 - x1**2 - x2**2 +a*(x1 + x2) + b*(x1**2 - x2)
fx1 = diff(f,x1)
fx2 = diff(f,x2)
result = solve([fx1,fx2,(x1**2-x2)*b,x1+x2],[x1,x2,a,b])

for i in range(len(result)):
    if result[i][3]>=0 and result[i][0]**2-result[i][1]<=0 and result[i][2]!=0:
        print(result[i])
        print("loss:",10 - result[i][0]**2 - result[i][1]**2 +result[i][2]*(result[i][1] + result[i][0]) +
              result[i][3]*(result[i][0]**2 - result[i][1]))




