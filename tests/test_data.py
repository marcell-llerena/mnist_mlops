"""Tests de datos: comprueban que el dataset tiene la forma esperada."""

import pytest
import torch

from mnist_mlops.config import CLASSES, EXAMPLES_DIR, N_TEST, N_TRAIN
from mnist_mlops.data import download_data, load_data, validate_data


def test_data_has_expected_size():
    train, test = load_data()
    assert len(train) == N_TRAIN
    assert len(test) == N_TEST


def test_images_are_28x28_and_between_0_and_1():
    train, _ = load_data()
    image, label = train[0]  # ya pasó por las transformaciones (tensor 1x28x28)
    assert image.shape == (1, 28, 28)
    assert 0 <= image.min() and image.max() <= 1
    assert 0 <= label <= 9


def test_data_has_ten_digits():
    train, _ = download_data()
    assert len(train.targets.unique()) == len(CLASSES)


def test_examples_exist_for_every_digit():
    for digit in CLASSES:
        assert (EXAMPLES_DIR / f"digito_{digit}.png").exists()


def test_validation_rejects_wrong_labels():
    _, test = download_data()
    test.targets = torch.full_like(test.targets, 15)  # etiqueta imposible: 15
    with pytest.raises(ValueError):
        validate_data(test, N_TEST)
