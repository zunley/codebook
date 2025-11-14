#/usr/bin/python3


class  Operator:
    def gradient(self, x):
        pass;

    def oper(self, x):
        pass ;


class SquareOperator:

    def gradient(self, x):
        return  2*x;
    def oper(self, x):
        return x*x

class ConstMultiOperator:
    def __init__(self, const):
        self.const = const

    def gradient(self, X):
        return self.const
        
    def oper(self, X):
        return self.const * X

class CompositeOperator(Operator):
    def __init__(self, outer: Operator, inner: Operator):
        self.outer = outer
        self.inner = inner

    def oper(self, X):
        inner_val = self.inner.oper(X)
        return self.outer.oper(inner_val)

    def gradient(self, X):
        inner_val = self.inner.oper(X)
        # 1. 外部梯度
        outer_grad = self.outer.gradient(inner_val)

        # 2. 内部梯度
        inner_grad = self.inner.gradient(X)

        return outer_grad * inner_grad

g = SquareOperator()
h = ConstMultiOperator(2)
f = CompositeOperator(h, g)

x = 9
delta = 0.01
epoch = 100

for i in range(epoch):
    loss = f.oper(x)
    print(f'%d: %d' %(i, loss))
    x = x - delta*f.gradient(x)

