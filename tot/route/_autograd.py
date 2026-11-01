"""Tiny numpy autodiff used by FlowDAGNN. Not a public API."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def as_tensor(value: Tensor | np.ndarray | float | int) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return Tensor(value)


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    g = np.asarray(grad, dtype=np.float64)
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and g.shape[axis] != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g.reshape(shape)


class Tensor:
    def __init__(self, data: np.ndarray | float | int, children: tuple[Tensor, ...] = ()) -> None:
        self.data = np.array(data, dtype=np.float64, copy=True)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._prev = children
        self._backward = lambda: None

    def backward(self) -> None:
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            nid = id(node)
            if nid in visited:
                return
            visited.add(nid)
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        for node in topo:
            node.grad = np.zeros_like(node.data, dtype=np.float64)
        self.grad = np.ones_like(self.data, dtype=np.float64)
        for node in reversed(topo):
            node._backward()

    def sum(self) -> Tensor:
        out = Tensor(np.sum(self.data), (self,))

        def _backward() -> None:
            self.grad += np.ones_like(self.data, dtype=np.float64) * out.grad

        out._backward = _backward
        return out

    def __add__(self, other: Tensor | float | int) -> Tensor:
        other = as_tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward() -> None:
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: float | int) -> Tensor:
        return self + other

    def __sub__(self, other: Tensor | float | int) -> Tensor:
        return self + (-as_tensor(other))

    def __rsub__(self, other: float | int) -> Tensor:
        return as_tensor(other) + (-self)

    def __neg__(self) -> Tensor:
        out = Tensor(-self.data, (self,))

        def _backward() -> None:
            self.grad -= out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Tensor | float | int) -> Tensor:
        other = as_tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def _backward() -> None:
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: float | int) -> Tensor:
        return self * other

    def __truediv__(self, other: Tensor | float | int) -> Tensor:
        other = as_tensor(other)
        inv = Tensor(1.0 / other.data, (other,))

        def _inv_backward() -> None:
            other.grad -= inv.grad / (other.data * other.data)

        inv._backward = _inv_backward
        return self * inv

    def __matmul__(self, other: Tensor) -> Tensor:
        other = as_tensor(other)
        out = Tensor(self.data @ other.data, (self, other))

        def _backward() -> None:
            a = self.data
            b = other.data
            g = out.grad
            if a.ndim == 0 or b.ndim == 0:
                raise ValueError("matmul expects at least 1d operands")
            if a.ndim == 1 and b.ndim == 1:
                self.grad += g * b
                other.grad += g * a
            elif a.ndim == 1 and b.ndim == 2:
                self.grad += b @ g
                other.grad += np.outer(a, g)
            elif a.ndim == 2 and b.ndim == 1:
                self.grad += np.outer(g, b)
                other.grad += a.T @ g
            else:
                self.grad += g @ b.T
                other.grad += a.T @ g

        out._backward = _backward
        return out

    def __getitem__(self, idx: int | slice) -> Tensor:
        out = Tensor(self.data[idx], (self,))

        def _backward() -> None:
            self.grad[idx] += out.grad

        out._backward = _backward
        return out


def tanh(tensor: Tensor | float | int) -> Tensor:
    tensor = as_tensor(tensor)
    y = np.tanh(tensor.data)
    out = Tensor(y, (tensor,))

    def _backward() -> None:
        tensor.grad += out.grad * (1.0 - y * y)

    out._backward = _backward
    return out


def sigmoid(tensor: Tensor | float | int) -> Tensor:
    tensor = as_tensor(tensor)
    x = np.clip(tensor.data, -60.0, 60.0)
    s = 1.0 / (1.0 + np.exp(-x))
    out = Tensor(s, (tensor,))

    def _backward() -> None:
        tensor.grad += out.grad * out.data * (1.0 - out.data)

    out._backward = _backward
    return out


def exp(tensor: Tensor | float | int) -> Tensor:
    tensor = as_tensor(tensor)
    y = np.exp(np.clip(tensor.data, -60.0, 60.0))
    out = Tensor(y, (tensor,))

    def _backward() -> None:
        tensor.grad += out.grad * out.data

    out._backward = _backward
    return out


def relu(tensor: Tensor | float | int) -> Tensor:
    tensor = as_tensor(tensor)
    mask = tensor.data > 0.0
    out = Tensor(np.where(mask, tensor.data, 0.0), (tensor,))

    def _backward() -> None:
        tensor.grad += out.grad * mask.astype(np.float64)

    out._backward = _backward
    return out


def abs_t(tensor: Tensor | float | int) -> Tensor:
    tensor = as_tensor(tensor)
    out = Tensor(np.abs(tensor.data), (tensor,))

    def _backward() -> None:
        tensor.grad += out.grad * np.sign(tensor.data)

    out._backward = _backward
    return out


def log1p(tensor: Tensor | float | int) -> Tensor:
    tensor = as_tensor(tensor)
    out = Tensor(np.log1p(tensor.data), (tensor,))

    def _backward() -> None:
        tensor.grad += out.grad / (1.0 + tensor.data)

    out._backward = _backward
    return out


def softmax(scores: Sequence[Tensor]) -> list[Tensor]:
    if not scores:
        return []
    shift = float(max(float(s.data) for s in scores))
    shifted = [exp(s - shift) for s in scores]
    total = shifted[0]
    for term in shifted[1:]:
        total = total + term
    return [term / total for term in shifted]


def softplus(tensor: Tensor | float | int) -> Tensor:
    """Stable ``log(1 + exp(x))``."""
    tensor = as_tensor(tensor)
    return relu(tensor) + log1p(exp(-abs_t(tensor)))


def bce_with_logits(logit: Tensor, target: float) -> Tensor:
    """Stable binary cross-entropy with a 0/1 target."""
    y = float(target)
    return relu(logit) - logit * y + log1p(exp(-abs_t(logit)))


def concat(tensors: Sequence[Tensor]) -> Tensor:
    parts = [as_tensor(t) for t in tensors]
    out = Tensor(np.concatenate([t.data.reshape(-1) for t in parts]), tuple(parts))
    sizes = [int(t.data.size) for t in parts]

    def _backward() -> None:
        offset = 0
        flat = np.asarray(out.grad, dtype=np.float64).reshape(-1)
        for tensor, size in zip(parts, sizes):
            tensor.grad += flat[offset : offset + size].reshape(tensor.data.shape)
            offset += size

    out._backward = _backward
    return out
