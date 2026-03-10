import sys

sys.path.append(".")
import numpy as np
from cma_es import CMAES


def test_cmaes_initialization():

    mean = np.zeros(5)
    sigma = 0.5

    cma = CMAES(mean, sigma)

    assert cma.dim == 5
    assert cma.population_size > 0
    assert cma.generation == 0


def test_sample_solution_dimension():

    mean = np.zeros(4)
    sigma = 1.0

    cma = CMAES(mean, sigma)

    x = cma.sample_solutions()

    assert isinstance(x, np.ndarray)
    assert x.shape == (4,)


def test_ask_returns_solution():

    mean = np.zeros(3)
    sigma = 1.0

    cma = CMAES(mean, sigma)

    x = cma.ask()

    assert len(x) == 3


def test_is_feasible():

    mean = np.zeros(2)
    sigma = 1.0

    bounds = np.array([[-1, 1], [-1, 1]])

    cma = CMAES(mean, sigma, bounds=bounds)

    x = np.array([0.5, 0.2])
    assert cma.is_feasible(x)

    x_bad = np.array([2, 0])
    assert not cma.is_feasible(x_bad)


def test_repair_infeasible():

    mean = np.zeros(2)
    sigma = 1

    bounds = np.array([[-1, 1], [-1, 1]])

    cma = CMAES(mean, sigma, bounds=bounds)

    x = np.array([2, -3])

    repaired = cma.repair_infeasible_params(x)

    assert repaired[0] <= 1
    assert repaired[1] >= -1


def test_tell_updates_generation():

    mean = np.zeros(3)
    sigma = 1

    cma = CMAES(mean, sigma)

    solutions = []

    for _ in range(cma.population_size):
        x = cma.ask()
        fitness = np.random.rand()
        solutions.append((x, fitness))

    cma.tell(solutions)

    assert cma.generation == 1


def test_eigen_decomposition_shapes():

    mean = np.zeros(4)
    sigma = 1

    cma = CMAES(mean, sigma)

    B, D = cma.eigen_decomposition()

    assert B.shape == (4, 4)
    assert D.shape == (4,)


def test_should_not_stop_initially():

    mean = np.zeros(3)
    sigma = 1

    cma = CMAES(mean, sigma)

    assert not cma.should_stop()


def test_reproducibility():

    mean = np.zeros(3)

    cma1 = CMAES(mean, 1, seed=42)
    cma2 = CMAES(mean, 1, seed=42)

    x1 = cma1.ask()
    x2 = cma2.ask()

    assert np.allclose(x1, x2)


def test_covariance_matrix_symmetric():

    mean = np.zeros(3)
    sigma = 1

    cma = CMAES(mean, sigma)

    solutions = []

    for _ in range(cma.population_size):
        x = cma.ask()
        solutions.append((x, np.random.rand()))

    cma.tell(solutions)

    C = cma._C

    assert np.allclose(C, C.T)
