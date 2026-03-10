import sys

sys.path.append(".")
import pytest
import torch
from PIL import Image
import numpy as np
from unittest.mock import patch
from torchvision import transforms
from torch.utils.data import TensorDataset
from datasets.get_dataset_with_transform import (
    CUTOUT,
    Lighting,
    Dataset2Class,
    get_datasets,
    get_nas_search_loaders,
)


def test_cutout_mask():

    img = torch.ones(3, 32, 32)

    cutout = CUTOUT(length=8)

    result = cutout(img.clone())

    assert result.shape == img.shape
    assert torch.sum(result) < torch.sum(img)


def test_cutout_bounds():

    img = torch.ones(3, 32, 32)

    cutout = CUTOUT(length=32)

    out = cutout(img)

    assert torch.min(out) >= 0


def test_lighting_output_shape():

    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

    img = Image.fromarray(img)

    aug = Lighting(alphastd=0.1)

    out = aug(img)

    assert out.size == img.size


def test_lighting_zero_alpha():

    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(img)

    aug = Lighting(alphastd=0)

    out = aug(img)

    assert np.array_equal(np.array(img), np.array(out))


def test_dataset_class_mapping():

    assert Dataset2Class["cifar10"] == 10
    assert Dataset2Class["cifar100"] == 100
    assert Dataset2Class["jaffe7"] == 7


class DummyDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


@patch("datasets.get_dataset_with_transform.dset.CIFAR10")
def test_get_cifar10_dataset(mock_cifar):

    mock_cifar.side_effect = [DummyDataset(50000), DummyDataset(10000)]  # train  # test

    train, test, xshape, classes = get_datasets("cifar10", root="data", cutout=-1)

    assert len(train) == 50000
    assert len(test) == 10000
    assert classes == 10
    assert xshape == (1, 3, 32, 32)


def test_transform_pipeline_output():

    transform = transforms.Compose(
        [
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )

    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(img)

    out = transform(img)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 48, 48)


def test_invalid_dataset():

    with pytest.raises(TypeError):

        get_datasets("invalid_dataset", root="data", cutout=-1)


def test_loader_split():

    x = torch.randn(100, 3, 32, 32)
    y = torch.randint(0, 10, (100,))

    dataset = TensorDataset(x, y)

    search, train, valid = get_nas_search_loaders(
        train_data=dataset,
        valid_data=dataset,
        dataset="jaffe7",
        config_root="configs",
        batch_size=8,
        workers=0,
    )

    assert train is not None
    assert valid is not None


def test_deterministic_split():

    gen = torch.Generator().manual_seed(42)

    x = torch.randn(50, 3, 32, 32)
    y = torch.randint(0, 7, (50,))
    dataset = TensorDataset(x, y)

    s1, t1, v1 = get_nas_search_loaders(
        dataset, dataset, "jaffe7", "configs", 4, 0, generator=gen
    )

    gen = torch.Generator().manual_seed(42)

    s2, t2, v2 = get_nas_search_loaders(
        dataset, dataset, "jaffe7", "configs", 4, 0, generator=gen
    )

    assert len(t1) == len(t2)
