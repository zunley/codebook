#!/usr/bin/env python3

import numpy as np

classes = 3
features = 5
num_train = 1

# 初始化一个参数向量

X = np.random.rand(num_train, features)

X = np.array([[1,2,3,4,5],
              [6,7,8,9,10],
              [11,12,13,14,15]])

y = np.array([0,1,2])

#W = np.random.rand(features, classes) #(5, 3)
#W = np.array([[0.53943785, 0.94824941],
#              [0.41194912, 0.21920829],
#              [0.15692327, 0.49342796],
#              [0.48345052, 0.35667389],
#              [0.13008523, 0.32586177]]
#)
W = np.array([[0.09643463, 0.25895286, 0.93194875],
              [0.49275667, 0.73000499, 0.30484739],
              [0.72099491, 0.28362841, 0.49700924],
              [0.23522966, 0.61365352, 0.4518322 ],
              [0.44349503, 0.89558813, 0.14241234]]
)

print(W)
delta = 1.0

f1 = X.dot(W)

maskx=np.arange(X.shape[0])
masky=y

sy = f1[maskx, masky][:, np.newaxis]
f1 = f1-sy + delta

f1[maskx, masky] = 0

#f1 = np.maximum(0, f1)
print(f1)
print(sy)

loss = np.sum(f1)


print("loss: %f" % (loss))

dw_bool =f1>0.astype(int)

print(dw_bool)
