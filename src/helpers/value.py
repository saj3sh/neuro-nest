import math
class Value:
    def __init__(self, data, children = (), label=''):
        self.data = data
        self.prev = set(children)
        self.label = label
        self.grad = 0.0
        self.backward = lambda:None
    def __repr__(self):
        return f'Value(data={self.data}, ∇ = {self.grad})'
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def back_handler():
            self.grad += out.grad
            other.grad += out.grad
        out.backward = back_handler
        return out
    def __radd__(self, other):
        return self + other
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def back_handler():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out.backward = back_handler
        return out
    def __rmul__(self, other):
        return self * other
    def __pow__(self, value):
        assert isinstance(value, (float, int)), "Only int and float values are supported."
        out = Value(self.data ** value, (self, ))
        def back_handler():
            self.grad +=  out.grad * (value * self.data ** (value - 1))
        out.backward = back_handler
        return out
    def relu(self):
        out = Value(max(self.data, 0), (self,))
        def back_handler():
            self.grad += 0 if out.data < 0 else out.grad
        out.backward = back_handler
        return out
    def tanh(self):
        t = (math.exp(self.data) - math.exp(-self.data))/(math.exp(self.data) + math.exp(-self.data))
        out = Value(t, (self, ))
        def back_handler():
            self.grad += out.grad * (1 - t ** 2)
        out.backward = back_handler
    def backward_pass(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
            topo.append(v)
        build_topo(self)
        self.grad = 1
        for value in reversed(topo):
            value.backward()


