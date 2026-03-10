import sys

sys.path.append("s1")
import pytest
import torch
from model_search import MixedOp, Cell, Network
from genotypes import PRIMITIVES


# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture
def dummy_input():
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def network():
    device = "cpu"
    return Network(C=16, num_classes=7, layers=2, device=device)


# -----------------------------
# MixedOp tests
# -----------------------------


def test_mixedop_forward():
    C = 16
    stride = 1

    op = MixedOp(C, stride)

    x = torch.randn(2, C, 32, 32)
    weights = torch.ones(len(PRIMITIVES))

    out = op(x, weights)

    assert out.shape[0] == 2
    assert out.shape[1] == C


# -----------------------------
# Cell tests
# -----------------------------


def test_cell_forward():
    steps = 4
    multiplier = 4

    cell = Cell(
        steps,
        multiplier,
        C_prev_prev=16,
        C_prev=16,
        C=16,
        reduction=False,
        reduction_prev=False,
    )

    s0 = torch.randn(2, 16, 32, 32)
    s1 = torch.randn(2, 16, 32, 32)

    k = sum(1 for i in range(steps) for _ in range(2 + i))
    weights = torch.randn(k, len(PRIMITIVES))

    out = cell(s0, s1, weights)

    assert out.shape[0] == 2
    assert out.shape[1] == multiplier * 16


# -----------------------------
# Network tests
# -----------------------------


def test_network_forward(network, dummy_input):

    logits = network(dummy_input)

    assert logits.shape[0] == 2
    assert logits.shape[1] == network._num_classes


# -----------------------------
# Architecture parameters tests
# -----------------------------


def test_random_alphas(network):

    alphas = network.random_alphas()

    assert len(alphas) == 2
    assert alphas[0].shape[0] == network._k


def test_update_and_check_alphas(network):

    new_alphas = network.random_alphas()

    network.update_alphas(new_alphas)

    assert network.check_alphas(new_alphas)


# -----------------------------
# Parametrized test
# -----------------------------


@pytest.mark.parametrize("discrete", [True, False])
def test_random_alphas_modes(network, discrete):

    alphas = network.random_alphas(discrete=discrete)

    assert len(alphas) == 2


# -----------------------------
# Genotype generation
# -----------------------------


def test_genotype_generation(network):

    genotype = network.genotype()

    assert genotype is not None
    assert hasattr(genotype, "normal")
    assert hasattr(genotype, "reduce")


# -----------------------------
# Dictionary preparation
# -----------------------------


def test_prepare_dicts(network):

    network.prepare_dicts()

    assert isinstance(network._nodes_dict, dict)
    assert isinstance(network._rows_dict, dict)
