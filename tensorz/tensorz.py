from __future__ import annotations
from typing import Iterable

import numpy as np

class TensorZ:
    def __init__(
        self,
        data: np.ndarray,
        children: Iterable[TensorZ] = (),
    ) -> None:
        self.data = data
        self.grad = np.zeros_like(data)
        raise NotImplementedError()

    def __add__(self, other: TensorZ) -> TensorZ:
        raise NotImplementedError()

    def __mul__(self, other: TensorZ) -> TensorZ:
        raise NotImplementedError()

    def __pow__(self, other: TensorZ) -> TensorZ:
        raise NotImplementedError()

    def __neg__(self) -> TensorZ:
        raise NotImplementedError()

    def __sub__(self, other: TensorZ) -> TensorZ:
        raise NotImplementedError()

    def __truediv__(self, other: TensorZ) -> TensorZ:
        raise NotImplementedError()

    def __repr__(self) -> str:
        raise NotImplementedError()

    def backward(self) -> None:
        raise NotImplementedError()
