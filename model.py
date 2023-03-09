import numpy as np
import scipy.linalg as linalg


class PartiallyInvertibleOperation:
    def transform(self, x):
        # should return `f(x), info`
        # where f(x) is a function of f,
        # info is a dict containing information to invert the function
        pass

    def invert_transform(self, fx, info):
        pass  # should revert the initial operation, i.e., `invert_transform(*transform(x)) = x`


class Cos(PartiallyInvertibleOperation):
    # Cosine is not entirely invertible, but here, we divide it into an invertible
    # and a non-invertible part in such a way that if called in sequence `invert_transform(*transform(x)) = x`
    def transform(self, angle):
        main_component = angle % (2 * np.pi)
        k = angle // (2 * np.pi)
        p = np.where((main_component > np.pi), 1, 0)
        return np.cos(main_component), {'k': k * 2 * np.pi, 'p': p}

    def invert_transform(self, fx, info):
        fx_bounded = np.minimum(np.maximum(fx, -1), 1)
        angle = np.where(info['p'], 2 * np.pi - np.arccos(fx_bounded), np.arccos(fx_bounded))
        return angle + info['k']


class Sin(PartiallyInvertibleOperation):
    def __init__(self):
        # same as Cos but shifted by pi/2
        self.cos = Cos()

    def transform(self, angle):
        return self.cos.transform(np.pi / 2 - angle)

    def invert_transform(self, fx, info):
        angle_aux = self.cos.invert_transform(fx, info)
        return np.pi / 2 - angle_aux


class AffineTransform(PartiallyInvertibleOperation):
    def __init__(self, A, b, regularization=0):
        self.A = A
        self.b = b
        self.regularization = regularization
        self.m, self.n = A.shape
        if self.m < self.n:
            raise ValueError('Only works whe A raises the dimension of the input vector')
        m, = b.shape
        if self.m != m:
            raise ValueError('b has the wrong dimension')

        if regularization > 0:
            # Ridge regularization
            self.A_pinv = np.linalg.inv(self.A.T @ self.A + regularization * np.eye(self.n)) @ self.A.T

        else:
            # Precompute factorization, so it can efficiently compute least square solution afterwards
            self.u, self.s, self.vh = linalg.svd(A, full_matrices=False, compute_uv=True)

    def transform(self, x):
        return x @ self.A.T + self.b, None

    def invert_transform(self, fx, _info):
        # Solve least square problem x = ||A @ x - (fx-b)|| , i.e., compute x = pinv(A) @ (fx - b)
        y = fx - self.b
        if self.regularization > 0:
            x = y @ self.A_pinv.T
        else:
            x = ((1 / self.s) * (y @ self.u)) @ self.vh  # == pinv(self.A)
        return x


class RBFSampler(PartiallyInvertibleOperation):
    """Invertible version of RBF sampler from sklearn"""

    def __init__(self, n_features, gamma=1.0, n_components=100, random_state=None, regularization=0):
        rng = np.random.RandomState(random_state)
        random_weights_ = np.sqrt(2 * gamma) * rng.normal(size=(n_components, n_features))
        random_offset_ = rng.uniform(0, 2 * np.pi, size=n_components)
        self.aff = AffineTransform(random_weights_, random_offset_, regularization)
        self.cos = Cos()
        self.n_features = n_features
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self.regularization = regularization

    def transform(self, x):
        x1, _ = self.aff.transform(x)
        z, info = self.cos.transform(x1)
        return np.sqrt(2.0) / np.sqrt(self.n_components) * z, info

    def invert_transform(self, f_x, info):
        f_x_rescaled = np.sqrt(self.n_components) / np.sqrt(2.0) * f_x
        f_x1 = self.cos.invert_transform(f_x_rescaled, info)
        x = self.aff.invert_transform(f_x1, None)
        return x
