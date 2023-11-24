from __future__ import annotations
from typing import Iterable, Union, Tuple, Optional, List

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
        assert self.data.size == 1
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

    def __repr__(self) -> str:
        return f"TensorZ: {self.data}"

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @staticmethod
    def zeros(dim: Union[Tuple[int], int]) -> TensorZ:
        return TensorZ(np.zeros(dim))

    @staticmethod
    def ones(dim: Union[Tuple[int], int]) -> TensorZ:
        return TensorZ(np.ones(dim))

    @staticmethod
    def normal(
        dim: Union[Tuple[int], int], mean: float = 0.0, std: float = 0.02
    ) -> TensorZ:
        return TensorZ(np.random.normal(mean, std, dim))

    @staticmethod
    def zeros_like(tensor: TensorZ) -> TensorZ:
        return TensorZ(np.zeros_like(tensor.data))

    @staticmethod
    def ones_like(tensor: TensorZ) -> TensorZ:
        return TensorZ(np.ones_like(tensor.data))

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
        out = TensorZ(self.data**other.data, (self, other))

        def _backward():
            self.grad += out.grad * other.data * (self ** (other - 1)).data
            other.grad += out.grad * out.data * self.log().data

        out._backward = _backward
        return out

    def __neg__(self) -> TensorZ:
        return self * -1

    def __sub__(self, other: Union[TensorZ, float, int]) -> TensorZ:
        return self + (other * -1)

    def __truediv__(self, other: Union[TensorZ, float, int]) -> TensorZ:
        return self * (other**-1)

    def log(self) -> TensorZ:
        out = TensorZ(np.log(self.data), (self,))

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def __radd__(self, other: Union[float, int]) -> TensorZ:
        return self + other

    def __rsub__(self, other: Union[float, int]) -> TensorZ:
        return self - other

    def __rmul__(self, other: Union[float, int]) -> TensorZ:
        return self * other

    def __rtruediv__(self, other: Union[float, int]) -> TensorZ:
        return other * (self**-1)

    def __matmul__(self, other: TensorZ) -> TensorZ:
        raise NotImplementedError()

    def where(self, condition: TensorZ, other: TensorZ) -> TensorZ:
        raise NotImplementedError()

    def cat(self, other: List[TensorZ], dim: int = -1) -> TensorZ:
        raise NotImplementedError()

    def sqrt(self) -> TensorZ:
        return self**0.5

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> TensorZ:
        out = TensorZ(self.data.mean(axis=axis, keepdims=keepdims), (self,))

        def _backward():
            if axis is not None:
                self.grad += np.expand_dims(out.grad, axis) / self.data.shape[axis]
            else:
                self.grad += out.grad / self.grad.size

        out._backward = _backward
        return out

    def var(self, axis: Optional[int] = None, keepdims: bool = False) -> TensorZ:
        raise NotImplementedError()

    def std(
        self,
        axis: Optional[int] = None,
        keepdims: bool = False,
        eps: Optional[float] = None,
    ) -> TensorZ:
        raise NotImplementedError()

    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> TensorZ:
        out = TensorZ(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _backward():
            if axis is not None and not keepdims:
                self.grad += np.expand_dims(out.grad, axis)
            else:
                self.grad += out.grad

        out._backward = _backward
        return out
