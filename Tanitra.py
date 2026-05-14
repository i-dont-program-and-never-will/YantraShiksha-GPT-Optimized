import numpy as np
from scipy.signal import convolve2d
from numpy.lib.stride_tricks import sliding_window_view


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def unbroadcast(grad, shape):
    """
    Reduce broadcasted gradients back to original shape.
    """
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


# ------------------------------------------------------------
# Main Tensor Class
# ------------------------------------------------------------

class Tanitra:
    __slots__ = ("data", "grad", "parents", "track_gradient", "shape")

    def __init__(self, data, track_gradient=True):
        self.data = np.asarray(data, dtype=np.float32)
        self.track_gradient = track_gradient
        self.parents = []
        self.shape = self.data.shape
        self.grad = None

    # --------------------------------------------------------
    # Addition
    # --------------------------------------------------------

    def __add__(self, other):
        if not isinstance(other, Tanitra):
            other = Tanitra(other, track_gradient=False)

        out = Tanitra(
            self.data + other.data,
            track_gradient=self.track_gradient or other.track_gradient
        )

        if self.track_gradient:
            out.parents.append(
                (self, lambda g: unbroadcast(g, self.shape))
            )

        if other.track_gradient:
            out.parents.append(
                (other, lambda g: unbroadcast(g, other.shape))
            )

        return out

    # --------------------------------------------------------
    # Subtraction
    # --------------------------------------------------------

    def __sub__(self, other):
        if not isinstance(other, Tanitra):
            other = Tanitra(other, track_gradient=False)

        out = Tanitra(
            self.data - other.data,
            track_gradient=self.track_gradient or other.track_gradient
        )

        if self.track_gradient:
            out.parents.append(
                (self, lambda g: unbroadcast(g, self.shape))
            )

        if other.track_gradient:
            out.parents.append(
                (other, lambda g: -unbroadcast(g, other.shape))
            )

        return out

    # --------------------------------------------------------
    # Multiplication
    # --------------------------------------------------------

    def __mul__(self, other):
        if not isinstance(other, Tanitra):
            other = Tanitra(other, track_gradient=False)

        out = Tanitra(
            self.data * other.data,
            track_gradient=self.track_gradient or other.track_gradient
        )

        if self.track_gradient:
            out.parents.append(
                (self, lambda g: unbroadcast(g * other.data, self.shape))
            )

        if other.track_gradient:
            out.parents.append(
                (other, lambda g: unbroadcast(g * self.data, other.shape))
            )

        return out

    # --------------------------------------------------------
    # Division
    # --------------------------------------------------------

    def __truediv__(self, other):
        if not isinstance(other, Tanitra):
            other = Tanitra(other, track_gradient=False)

        out = Tanitra(
            self.data / other.data,
            track_gradient=self.track_gradient or other.track_gradient
        )

        if self.track_gradient:
            out.parents.append(
                (
                    self,
                    lambda g: unbroadcast(g / other.data, self.shape)
                )
            )

        if other.track_gradient:
            out.parents.append(
                (
                    other,
                    lambda g: unbroadcast(
                        -g * self.data / (other.data ** 2),
                        other.shape
                    )
                )
            )

        return out

    # --------------------------------------------------------
    # Matrix Multiplication
    # --------------------------------------------------------

    def __matmul__(self, other):
        if not isinstance(other, Tanitra):
            other = Tanitra(other, track_gradient=False)

        out = Tanitra(
            self.data @ other.data,
            track_gradient=self.track_gradient or other.track_gradient
        )

        if self.track_gradient:
            out.parents.append(
                (self, lambda g: g @ other.data.T)
            )

        if other.track_gradient:
            out.parents.append(
                (other, lambda g: self.data.T @ g)
            )

        return out

    # --------------------------------------------------------
    # Indexing
    # --------------------------------------------------------

    def __getitem__(self, index):
        out = Tanitra(
            self.data[index],
            track_gradient=self.track_gradient
        )

        if self.track_gradient:

            def grad_fn(g):
                grad = np.zeros_like(self.data)
                grad[index] += g
                return grad

            out.parents.append((self, grad_fn))

        return out

    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    def flatten(self):
        out = Tanitra(
            self.data.flatten(),
            track_gradient=self.track_gradient
        )

        if self.track_gradient:
            out.parents.append(
                (self, lambda g: g.reshape(self.shape))
            )

        return out

    # --------------------------------------------------------
    # Transpose
    # --------------------------------------------------------

    @property
    def T(self):
        out = Tanitra(
            self.data.T,
            track_gradient=self.track_gradient
        )

        if self.track_gradient:
            out.parents.append((self, lambda g: g.T))

        return out

    # --------------------------------------------------------
    # Dot Product
    # --------------------------------------------------------

    def dot(self, other):
        if not isinstance(other, Tanitra):
            other = Tanitra(other, track_gradient=False)

        out = Tanitra(
            np.dot(self.data, other.data),
            track_gradient=self.track_gradient or other.track_gradient
        )

        if self.track_gradient:
            out.parents.append(
                (self, lambda g: g * other.data)
            )

        if other.track_gradient:
            out.parents.append(
                (other, lambda g: g * self.data)
            )

        return out

    # --------------------------------------------------------
    # Backward Pass (Topological)
    # --------------------------------------------------------

    def backward(self, grad=None):

        if grad is None:
            grad = np.ones_like(self.data, dtype=np.float32)

        topo = []
        visited = set()

        def build(node):
            if id(node) not in visited:
                visited.add(id(node))

                for parent, _ in node.parents:
                    build(parent)

                topo.append(node)

        build(self)

        self.grad = grad

        for node in reversed(topo):

            if node.grad is None:
                continue

            for parent, grad_fn in node.parents:

                g = grad_fn(node.grad)

                if parent.grad is None:
                    parent.grad = g
                else:
                    parent.grad += g

    # --------------------------------------------------------
    # Zero Grad
    # --------------------------------------------------------

    def zero_grad(self):

        topo = []
        visited = set()

        def build(node):
            if id(node) not in visited:
                visited.add(id(node))

                for parent, _ in node.parents:
                    build(parent)

                topo.append(node)

        build(self)

        for node in topo:
            node.grad = None

    # --------------------------------------------------------
    # Representation
    # --------------------------------------------------------

    def __repr__(self):
        return f"Tanitra(data={self.data}, grad={self.grad})"


# ------------------------------------------------------------
# Activation Functions
# ------------------------------------------------------------

def sigmoid(x):

    if not isinstance(x, Tanitra):
        x = Tanitra(x)

    sig = 1 / (1 + np.exp(-x.data))

    out = Tanitra(sig, track_gradient=x.track_gradient)

    if x.track_gradient:
        out.parents.append(
            (x, lambda g: g * sig * (1 - sig))
        )

    return out


def relu(x):

    if not isinstance(x, Tanitra):
        x = Tanitra(x)

    result = np.maximum(0, x.data)

    out = Tanitra(result, track_gradient=x.track_gradient)

    if x.track_gradient:
        out.parents.append(
            (
                x,
                lambda g: g * (x.data > 0).astype(np.float32)
            )
        )

    return out


def tanh(x):

    if not isinstance(x, Tanitra):
        x = Tanitra(x)

    t = np.tanh(x.data)

    out = Tanitra(t, track_gradient=x.track_gradient)

    if x.track_gradient:
        out.parents.append(
            (x, lambda g: g * (1 - t ** 2))
        )

    return out


def softmax(x, axis=-1):

    if not isinstance(x, Tanitra):
        x = Tanitra(x)

    shifted = x.data - np.max(x.data, axis=axis, keepdims=True)

    exp = np.exp(shifted)

    result = exp / np.sum(exp, axis=axis, keepdims=True)

    out = Tanitra(result, track_gradient=x.track_gradient)

    if x.track_gradient:

        def grad_fn(g):
            return g * result * (1 - result)

        out.parents.append((x, grad_fn))

    return out


# ------------------------------------------------------------
# Math Functions
# ------------------------------------------------------------

def square(x):

    if not isinstance(x, Tanitra):
        x = Tanitra(x)

    out = Tanitra(
        np.square(x.data),
        track_gradient=x.track_gradient
    )

    if x.track_gradient:
        out.parents.append(
            (x, lambda g: 2 * x.data * g)
        )

    return out


def mean(x, axis=None):

    if not isinstance(x, Tanitra):
        x = Tanitra(x)

    result = np.mean(x.data, axis=axis)

    out = Tanitra(result, track_gradient=x.track_gradient)

    if x.track_gradient:

        divisor = np.prod(x.data.shape) if axis is None else x.data.shape[axis]

        def grad_fn(g):
            return np.ones_like(x.data) * g / divisor

        out.parents.append((x, grad_fn))

    return out


def log(x):

    if not isinstance(x, Tanitra):
        x = Tanitra(x)

    safe = np.clip(x.data, 1e-9, None)

    out = Tanitra(
        np.log(safe),
        track_gradient=x.track_gradient
    )

    if x.track_gradient:
        out.parents.append(
            (x, lambda g: g / safe)
        )

    return out


def sin(x):

    if not isinstance(x, Tanitra):
        x = Tanitra(x)

    out = Tanitra(
        np.sin(x.data),
        track_gradient=x.track_gradient
    )

    if x.track_gradient:
        out.parents.append(
            (x, lambda g: g * np.cos(x.data))
        )

    return out


def cos(x):

    if not isinstance(x, Tanitra):
        x = Tanitra(x)

    out = Tanitra(
        np.cos(x.data),
        track_gradient=x.track_gradient
    )

    if x.track_gradient:
        out.parents.append(
            (x, lambda g: -g * np.sin(x.data))
        )

    return out


# ------------------------------------------------------------
# Convolution
# ------------------------------------------------------------

def convolution2d(
        a,
        b,
        stride=1,
        padding_mode=None,
        pad_width=0,
        constant_values=0
):

    if padding_mode is not None:

        a_padded = np.pad(
            a.data,
            pad_width=pad_width,
            mode=padding_mode,
            constant_values=constant_values
        )

    else:
        a_padded = a.data

    kernel = np.flip(b.data)

    result = convolve2d(
        a_padded,
        kernel,
        mode='valid'
    )[::stride, ::stride]

    out = Tanitra(
        result,
        track_gradient=a.track_gradient or b.track_gradient
    )

    # --------------------------------------------------------
    # Gradient w.r.t input
    # --------------------------------------------------------

    if a.track_gradient:

        def grad_a(g):

            if stride > 1:

                expanded = np.zeros(
                    (
                        (g.shape[0] - 1) * stride + 1,
                        (g.shape[1] - 1) * stride + 1
                    ),
                    dtype=np.float32
                )

                expanded[::stride, ::stride] = g

                g = expanded

            grad = convolve2d(g, kernel, mode='full')

            if padding_mode is not None:

                if isinstance(pad_width, tuple):

                    grad = grad[
                        pad_width[0][0]: grad.shape[0] - pad_width[0][1],
                        pad_width[1][0]: grad.shape[1] - pad_width[1][1]
                    ]

                else:

                    grad = grad[
                        pad_width:-pad_width,
                        pad_width:-pad_width
                    ]

            return grad

        out.parents.append((a, grad_a))

    # --------------------------------------------------------
    # Gradient w.r.t kernel
    # --------------------------------------------------------

    if b.track_gradient:

        def grad_b(g):

            if stride > 1:

                expanded = np.zeros(
                    (
                        (g.shape[0] - 1) * stride + 1,
                        (g.shape[1] - 1) * stride + 1
                    ),
                    dtype=np.float32
                )

                expanded[::stride, ::stride] = g

                g = expanded

            return convolve2d(a_padded, g, mode='valid')

        out.parents.append((b, grad_b))

    return out


# ------------------------------------------------------------
# Max Pooling
# ------------------------------------------------------------

def pooling2d(a, pool_size=2, stride=2):

    windows = sliding_window_view(
        a.data,
        (pool_size, pool_size)
    )

    windows = windows[::stride, ::stride]

    out_data = windows.max(axis=(-1, -2))

    out = Tanitra(
        out_data,
        track_gradient=a.track_gradient
    )

    if a.track_gradient:

        max_indices = windows.reshape(
            *windows.shape[:2],
            -1
        ).argmax(axis=-1)

        def grad_fn(g):

            grad = np.zeros_like(a.data)

            h, w = g.shape

            for i in range(h):
                for j in range(w):

                    idx = max_indices[i, j]

                    r = idx // pool_size
                    c = idx % pool_size

                    grad[
                        i * stride + r,
                        j * stride + c
                    ] += g[i, j]

            return grad

        out.parents.append((a, grad_fn))

    return out


# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def to_numpy(x):
    return x.data


def length(x):
    return len(x.data)


# ------------------------------------------------------------
# Example
# ------------------------------------------------------------

if __name__ == "__main__":

    a = Tanitra([[1, 2], [3, 4]])
    b = Tanitra([[5, 6], [7, 8]])

    c = a @ b
    d = relu(c)
    e = mean(d)

    e.backward()

    print("Output:")
    print(e.data)

    print("\nGradients:")
    print("a.grad:")
    print(a.grad)

    print("\nb.grad:")
    print(b.grad)
