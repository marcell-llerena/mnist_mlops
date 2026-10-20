"""Preparación común para los tests: asegura que exista un modelo entrenado."""

import pytest

from mnist_mlops.config import MODEL_PATH
from mnist_mlops.train import main as train_main


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    if not MODEL_PATH.exists():
        train_main()
