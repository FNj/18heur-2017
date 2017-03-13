import numpy as np


class ObjFun(object):
    """Generic objective function super-class."""

    def __init__(self, fstar, a, b):
        """
        Default initialization function (for discrete tasks) that sets:
        * f* value to be reached (can be -inf)
        * a: lower bound vector
        * b: upper bound vector
        """
        self.fstar = fstar
        self.a = a
        self.b = b

    def get_fstar(self):
        return self.fstar

    def get_bounds(self):
        return [self.a, self.b]

    def generate_point(self):
        raise NotImplementedError("Objective function must implement its own random point generation")

    def get_neighborhood(self, x):
        return x

    def evaluate(self, x):
        raise NotImplementedError("Objective function must implement its own evaluation")


class AirShip(ObjFun):
    """1-dimensional demo task from the first excercise."""

    def __init__(self):
        fstar = -100
        a = 0
        b = 799
        super().__init__(fstar, a, b)

    def generate_point(self):
        return np.random.randint(0, 800)

    def get_neighborhood(self, x, d):
        left = [x for x in np.arange(x - 1, x - d - 1, -1, dtype=int) if x >= 0]
        right = [x for x in np.arange(x + 1, x + d + 1, dtype=int) if x < 800]
        if np.size(left) == 0:
            return right
        elif np.size(right) == 0:
            return left
        else:
            return np.concatenate((left, right))

    def evaluate(self, x):
        px = np.array([0, 50, 100, 300, 400, 700, 799], dtype=int)
        py = np.array([0, 100, 0, 0, 25, 0, 50], dtype=int)
        xx = np.arange(0, 800)
        yy = np.interp(xx, px, py)
        return -yy[x]  # negative altitude, becase we are minimizing (to be consistent with other obj. functions)


class Sum(ObjFun):
    def __init__(self, a, b):
        self.n = np.size(a)  # dimension of the task
        super().__init__(fstar=0, a=a, b=b)

    def generate_point(self):
        return [np.random.randint(self.a[i], self.b[i] + 1) for i in np.arange(self.n)]

    def get_neighborhood(self, x, d):
        assert d == 1, "Sum(x) supports neighbourhood with distance = 1 only"
        nd = []
        for i in np.arange(self.n):
            if x[i] > self.a[i]:
                xx = x.copy()
                xx[i] -= 1
                nd.append(xx)
            if x[i] < self.b[i]:
                xx = x.copy()
                xx[i] += 1
                nd.append(xx)
        return nd

    def evaluate(self, x):
        return np.sum(x)


class SumSinx(ObjFun):
    def __init__(self, a, b):
        self.n = np.size(a)  # dimension of the task
        super().__init__(fstar=0, a=a, b=b)

    def generate_point(self):
        return [np.random.randint(self.a[i], self.b[i] + 1) for i in np.arange(self.n)]

    def get_neighborhood(self, x, d):
        assert d == 1, "SumSinx(x) supports neighbourhood with distance = 1 only"
        nd = []
        for i in np.arange(self.n):
            if x[i] > self.a[i]:
                xx = x.copy()
                xx[i] -= 1
                nd.append(xx)
            if x[i] < self.b[i]:
                xx = x.copy()
                xx[i] += 1
                nd.append(xx)
        return nd

    def evaluate(self, x):
        return np.sum(np.abs(x * np.sin(x)))


class TSPGrid(ObjFun):
    def __init__(self, par_a, par_b, norm=2):
        n = par_a * par_b  # number of cities

        # compute city coordinates
        grid = np.zeros((n, 2), dtype=np.int)
        for i in np.arange(par_a):
            for j in np.arange(par_b):
                grid[i * par_b + j] = np.array([i, j])

        # compute distances based on coordinates
        dist = np.zeros((n, n))
        for i in np.arange(n):
            for j in np.arange(i + 1, n):
                dist[i, j] = np.linalg.norm(grid[i, :] - grid[j, :], norm)
                dist[j, i] = dist[i, j]

        self.fstar = n + np.mod(n, 2) * (2 ** (1 / norm) - 1)
        self.n = n
        self.dist = dist
        self.a = np.zeros(n - 1, dtype=np.int)  # n-1 because the first city is pre-determined
        self.b = np.arange(n - 2, -1, -1)

    def generate_point(self):
        return [np.random.randint(self.a[i], self.b[i] + 1) for i in np.arange(self.n - 1)]
        # return [np.random.randint(0, i+1) for i in self.a]

    def decode(self, x):
        """
        Decodes solution vector into ordered list of visited cities, e.g:
         x = 1 2 2 1 0
        cx = 2 4 5 3 1
        """
        cx = np.zeros(self.n, dtype=np.int)  # the final tour
        ux = np.ones(self.n, dtype=np.int)  # used cities indices
        ux[0] = 0  # first city is used automatically
        c = np.cumsum(ux)  # cities to be included in the tour
        for k in np.arange(1, self.n):
            ix = x[k - 1] + 1  # order index of currently visited city
            cc = c[ix]  # currently visited city
            cx[k] = cc  # append visited city into final tour
            c = np.delete(c, ix)  # visited city can not be included in the tour any more
        return cx

    def tour_dist(self, cx):
        d = 0
        for i in np.arange(self.n):
            dx = self.dist[cx[i - 1], cx[i]] if i > 0 else self.dist[cx[self.n - 1], cx[i]]
            d += dx
        return d

    def evaluate(self, x):
        cx = self.decode(x)
        return self.tour_dist(cx)

    def get_neighborhood(self, x, d):
        assert d == 1, "TSPGrid supports neighbourhood with distance = 1 only"
        nd = []
        for i, xi in enumerate(x):
            # x-lower
            if x[i] > self.a[i]:  # (!) mutation correction .. will be discussed later
                xl = x.copy()
                xl[i] = x[i] - 1
                nd.append(xl)

            # x-upper
            if x[i] < self.b[i]:  # (!) mutation correction ..  -- // --
                xu = x.copy()
                xu[i] = x[i] + 1
                nd.append(xu)

        return nd
