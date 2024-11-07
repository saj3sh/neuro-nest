from helpers.mlp import MLP
from helpers.value import Value

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets
NO_OF_EPOCH = 20

network =  MLP(3, [4, 4, 1])
for k in range(200):
    ypred = [network(xi) for xi in xs]
    loss: Value = sum((-y + yi) ** 2 for y, yi in zip(ys, ypred))
    for p in network.parameters():
        p.grad = 0.0
    loss.backward_pass()
    for p in network.parameters():
        p.data += -0.001 * p.grad
    print(k, loss.data)
print (ypred)
