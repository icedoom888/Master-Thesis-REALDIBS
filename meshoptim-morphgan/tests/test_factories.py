# -*- coding: utf-8 -*-
# Author: Joan Massich <joan.massich@disneyresearch.com>
#         Andreas Aumiller <andy.aumiller@disneyresearch.com>

import pytest
import tensorflow as tf

from noice import make_model, make_dataset, make_optimizer, make_loss
from noice.losses import registered_losses
from noice.datasets.base import registered_datasets
from noice.networks import registered_models
from noice.optimizers import registered_optimizers


@pytest.mark.parametrize(
    "make, factory_type",
    [
        pytest.param(make_model, "models", id="models"),
        pytest.param(make_dataset, "datasets", id="datasets"),
        pytest.param(make_optimizer, "optimizers", id="optimizers"),
        pytest.param(make_loss, "losses", id="foobar"),
    ],
)
def test_factory_missing_kind(make, factory_type):
    """Test factory error handling."""
    _msg = "Invalid value for kind .* {current} .* but got 'foo' instead"
    with pytest.raises(ValueError, match=_msg.format(current=factory_type)):
        make(kind="foo")


@pytest.mark.parametrize(
    "kind", [pytest.param(kind, id=kind) for kind in registered_losses],
)
def test_make_loss(kind):
    """Test make_loss method function creation."""
    assert callable(make_loss(kind=kind))


@pytest.mark.parametrize(
    "kind, expected_class",
    [
        pytest.param(kind, registered_models[kind], id=kind)
        for kind in registered_models
    ],
)
def test_make_model(kind, expected_class):
    """Test make_model method instance creation and takes required kwargs."""
    required_common_kwargs = {'num_classes': 42}
    assert isinstance(
        make_model(kind=kind, **required_common_kwargs),
        expected_class
    )


@pytest.mark.parametrize(
    "kind", [pytest.param(kind, id=kind) for kind in registered_datasets],
)
def test_make_dataset(kind):
    """Test make_dataset method creates a tf.data.Dataset."""
    assert isinstance(make_dataset(kind=kind), tf.data.Dataset)


@pytest.mark.parametrize(
    "kind, expected_class",
    [
        pytest.param(kind, registered_optimizers[kind], id=kind)
        for kind in registered_optimizers
    ],
)
def test_make_optimizer(kind, expected_class):
    """Test make_optimizer method instance creation."""
    assert isinstance(make_optimizer(kind=kind), registered_optimizers[kind])
