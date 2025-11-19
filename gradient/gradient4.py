class Value:
  def __init__(self, data, op=None, parents=(), name='unamed'):
    self.data = data
    self.grad = 0
    self.op = op
    self.parents = parents
    self.name = name
  
  def backward(self, grad = 1):
      self.grad += grad
      # 这是一个叶子节点
      if self.op is None:
          return
      grads = self.op.backward(self)
      
      for parent, grad in zip(self.parents, grads):
          parent.backward(grad)

class AddOperator:
  def forward(self, x,y):
    out = Value(x.data+y.data, op = self, parents=(x,y))
    return out
  
  def backward(self, out):
      x, y = out.parents
      return out.grad, out.grad
      
class MulOperator:
  def forward(self, x, y):
    out = Value(x.data*y.data, op=self, parents=(x,y))
    return out
    
  def backward(self, out):
    x, y = out.parents
    return out.grad*y.data, out.grad*x.data
  
  
# 构建一个 f(x) = (x+y)y
x = Value(5.0, name='x')
y = Value(2.0, name='y')

lr = 0.1
steps = 100

add = AddOperator()
mul = MulOperator()


for step in range(steps):
  u = add.forward(x, y)
  u.name = 'u'
  loss = mul.forward(u, y)
  loss.name = 'loss'

  

  loss.backward(1)
  
  # 更新参数
  x.data -= lr*x.grad
  y.data -= lr*y.grad
  

  # 重置梯度
  x.grad = 0
  y.grad = 0
  print(f"%d: %0.2f" %(step, loss.data))
