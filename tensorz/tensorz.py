from __future__ import annotations
from typing import Iterable, Union

import numpy as np


class TensorZ:
    def __init__(
        self,
        data: Union[np.ndarray, float, int],
        _children: Iterable[TensorZ] = (),
    ) -> None:
        data = data if isinstance(data, np.ndarray) else np.array([data])
        self.data = data.astype(float)
        self.grad = np.zeros_like(data).astype(float)
        self._backward = lambda: None
        self._prev = _children


    def backward(self) -> None:
        # create topo from DAG
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build_topo(p)
                topo.append(v)
        build_topo(self)

        # backprop grads
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
            

    def __add__(self, other: Union[TensorZ, float, int]) -> TensorZ:
        other = other if isinstance(other, TensorZ) else TensorZ(other)
        out = TensorZ(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out


    def __mul__(self, other: Union[TensorZ, float, int]) -> TensorZ:
        other = other if isinstance(other, TensorZ) else TensorZ(other)
        out = TensorZ(self.data * other.data, (self, other))

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out


    def __pow__(self, other: Union[TensorZ, float, int]) -> TensorZ:
        other = other if isinstance(other, TensorZ) else TensorZ(other)
        out = TensorZ(self.data ** other.data, (self, other))

        def _backward():
            self.grad += out.grad * other.data * (self ** (other - 1)).data
            other.grad += out.grad * out.data * self.ln().data

        out._backward = _backward
        return out


    def __neg__(self) -> TensorZ:
        return self * -1


    def __sub__(self, other: Union[TensorZ, float, int]) -> TensorZ:
        return self + (other * -1)


    def __truediv__(self, other: Union[TensorZ, float, int]) -> TensorZ:
        return self * (other ** -1)


    def __repr__(self) -> str:
        return f'TensorZ: {self.data}'


    def ln(self) -> TensorZ:
        out = TensorZ(np.log(self.data), (self,))

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

