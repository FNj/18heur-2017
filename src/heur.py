import numpy as np


class StopCriterion(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Heuristic:
    def __init__(self, of, maxeval):
        self.of = of
        self.maxeval = maxeval
        self.best_y = np.inf
        self.best_x = None
        self.neval = 0

    def evaluate(self, x):
        y = self.of.evaluate(x)
        self.neval += 1
        if y < self.best_y:
            self.best_y = y
            self.best_x = x
        if y <= self.of.get_fstar():
            raise StopCriterion('Found solution with desired fstar value')
        if self.neval == self.maxeval:
            raise StopCriterion('Exhausted maximum allowed number of evaluations')
        return y

    def report_end(self):
        return {
            'best_y': self.best_y,
            'best_x': self.best_x,
            'neval': self.neval if self.best_y <= self.of.get_fstar() else np.inf
        }


class ShootAndGo(Heuristic):
    def __init__(self, of, maxeval, hmax=np.inf, random_descent=False):
        Heuristic.__init__(self, of, maxeval)
        self.hmax = hmax
        self.random_descent = random_descent

    def steepest_descent(self, x):
        # Steepest/Random Hill Descent beginning in x
        desc_best_y = np.inf
        desc_best_x = x
        h = 0
        go = True
        while go and h < self.hmax:
            go = False
            h += 1

            neighborhood = self.of.get_neighborhood(desc_best_x, 1)
            if self.random_descent:
                np.random.shuffle(neighborhood)

            for xn in neighborhood:
                yn = self.evaluate(xn)
                if yn < desc_best_y:
                    desc_best_y = yn
                    desc_best_x = xn
                    go = True
                    if self.random_descent:
                        break

    def search(self):
        try:
            while True:
                # Random Shoot...
                x = self.of.generate_point()  # global search
                self.evaluate(x)
                # ...and Go (optional)
                if self.hmax > 0:
                    self.steepest_descent(x)  # local search

        except StopCriterion:
            return self.report_end()
        except:
            raise


class SimulatedAnnealing(Heuristic):
    def __init__(self, of, maxeval, max_temperature=1e4, cooling_rate=3e-3, min_temperature=1, restart_period=np.inf):
        Heuristic.__init__(self, of, maxeval)
        assert max_temperature > 0, "Maximum temperature must be positive."
        self.starting_temperature = max_temperature
        assert cooling_rate > 0 and cooling_rate < 1, "Cooling rate must be between 0 and 1."
        self.cooling_coefficient = 1 - cooling_rate
        assert min_temperature > 0 and min_temperature < max_temperature, "Minimum temperature must be positive" \
                                                                          " and smaller than maximum temperature."
        self.min_temperature = min_temperature
        self.max_exponent = (np.log(self.min_temperature) - np.log(self.starting_temperature)) / np.log(
            self.cooling_coefficient)
        assert restart_period % 1 == 0 or restart_period == np.inf, "Restart period must be integral."
        self.restart_period = restart_period
        self.steps = 1

    def search(self):
        try:
            x = self.of.generate_point()
            y = self.evaluate(x)
            while True:
                if self.steps % self.restart_period == 0:
                    x = self.best_x
                    y = self.best_y
                self.temperature = self.starting_temperature * self.cooling_coefficient ** (
                    self.neval / self.maxeval * self.max_exponent)  # cool down
                neighborhood = self.of.get_neighborhood(x, 1)
                rand_index = np.random.randint(len(neighborhood))
                candidate_x = neighborhood[rand_index]
                candidate_y = self.evaluate(candidate_x)
                if np.exp((y - candidate_y) / self.temperature) > np.random.rand():
                    x = candidate_x
                    y = candidate_y
                self.steps += 1
        except StopCriterion:
            return self.report_end()
        except:
            raise
