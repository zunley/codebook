#/usr/bin/python3

class  Operator:
    def gradient(self, x):
        pass;

    def oper(self, x):
        pass ;


class MulOperator:
    def oper(self, x, y):
        return x*y

    def gradient(self, x, y):
        return y, x

x = 5
y = 5
lr = 0.1
steps = 100

f = MulOperator()

for step in range(steps):
    loss = f.oper(x,y)
    print(f'%d: %d' %(step, loss))
    grad_x, grad_y = f.gradient(x, y)
    x = x - lr * grad_x
    y = y - lr*grad_y