# -*- coding: utf-8 -*-
# Author: Hayko Riemenschneider <hayko@disneyresearch.com>

import pytest
import tensorflow as tf
from inspect import signature
from numpy.testing import assert_array_equal

from noice.losses import registered_losses
from noice.losses import norm_crossentropy
from noice.losses import softmax_crossentropy
from noice.losses import norm_crossentropy_mean
from noice.losses import softmax_crossentropy_mean


@pytest.mark.parametrize(
    "loss_func",
    [pytest.param(loss, id=kind) for kind, loss in registered_losses.items()],
)
def test_loss_function_signature(loss_func):
    """Test common required parameters present in signature in right order."""
    REQUIRED_PARAMETERS = ['y_true', 'y_pred']
    _first_two_parameters = list(signature(loss_func).parameters.keys())[:2]
    assert _first_two_parameters == REQUIRED_PARAMETERS


@pytest.mark.parametrize(
    "first_func, second_func",
    [
        pytest.param(norm_crossentropy_mean, norm_crossentropy, id='cross entropy'),
        pytest.param(softmax_crossentropy_mean, softmax_crossentropy, id='softmax'),
    ],
)
@pytest.mark.parametrize(
    "_params",
    [
        pytest.param(
            dict(y_pred=tf.convert_to_tensor([1, 0, 0, 0], tf.float32),
                 y_true=tf.convert_to_tensor([1, 1, 0, 0], tf.float32)),
            id='1D',
        ),
        pytest.param(
            dict(y_pred=tf.convert_to_tensor([[1, 0, 0, 0], [1, 0, 0, 0]], tf.float32),
                 y_true=tf.convert_to_tensor([[1, 1, 0, 0], [1, 1, 0, 0]], tf.float32)),
            id='2D',
        )
    ],
)
def test_equivalent_pairs_of_loss_functions(first_func, second_func, _params):
    """Test difference between pairs of equivalent loss functions."""
    assert_array_equal(first_func(**_params), second_func(**_params))
