# -*- coding: utf-8 -*-
# Author: Hayko Riemenschneider <hayko@disneyresearch.com>
#         Joan Massich <joan.massich@disneyresearch.com>

import pytest
import tensorflow as tf

from noice import make_model, make_dataset, make_optimizer, make_loss
from noice.utils import Bunch, BunchConst
from noice.base import setup_training, get_current_experiment_flags, expand_config
from noice._flags import dataset_flags


optimizer_flags = Bunch(
    kind='adam',
    lr=2e-4,  # learning_rate
    epsilon=1e-6,  # adam_eps
)


@pytest.mark.skip(reason="this thakes forever")
def test_single_experiment():
    """Test the mnist experiment (smoky test to check coverage)."""
    _ = [
        tf.config.experimental.set_memory_growth(gpu, True)
        for gpu in tf.config.experimental.list_physical_devices('GPU')
    ]
    setup_training(
        nb_epoch=1,
        dataset=make_dataset(**dataset_flags),
        model=make_model(kind='my_model', num_classes=10),
        optimizer=make_optimizer(**optimizer_flags),
        loss_fn=make_loss('softce'),
    )


def test_call_example_with_flags():
    """Test iterate over multiple experiments where each config is fully specified."""
    def setup_training(
        *, nb_epoch=None, dataset=None, model=None, optimizer=None, loss_fn=None
    ):
        pass

    for dataset, model, optimizer, loss_kind in get_current_experiment_flags(
        datasets=[
            Bunch(kind="mnist")
        ],
        models=[
            Bunch(kind='my_model', num_classes=10)
        ],
        optimizers=[
            Bunch(kind='adam', lr=1e-6),
            Bunch(kind='adam', lr=1e-3),
            Bunch(kind='adam', lr=1e-1),

            Bunch(kind='sgd', lr=1e-6),
            Bunch(kind='sgd', lr=1e-3),
            Bunch(kind='sgd', lr=1e-1),
        ],
        loss_names=[
            'softce',
            'softce',
        ]
        # epoch=[5],
    ):
        assert dataset.kind == 'mnist'
        assert model.kind == 'my_model'
        assert isinstance(optimizer.kind, str)
        assert optimizer.kind in ['adam', 'sgd']

        setup_training(
            nb_epoch=42,
            dataset=make_dataset(**dataset),
            model=make_model(**model),
            optimizer=make_optimizer(**optimizer),
            loss_fn=make_loss(loss_kind),
        )


def test_expand_config():
    """Test genearating different configurations by giving an iterator to the param."""
    EXPECTED = [
        BunchConst(foo='a', bar='x', baz='baz'),
        BunchConst(foo='a', bar='y', baz='baz'),
        BunchConst(foo='b', bar='x', baz='baz'),
        BunchConst(foo='b', bar='y', baz='baz'),
    ]

    actual = expand_config(
        base=dict(foo='foo'), foo=['a', 'b'], bar=['x', 'y'], baz=['baz']
    )

    for aa, ee in zip(actual, EXPECTED):
        assert isinstance(aa, dict)
        assert aa == ee


def test_combining_experiments():
    """Test combining the configuration expansion and the experiment generation."""
    def setup_training(
        *, nb_epoch=None, dataset=None, model=None, optimizer=None, loss_fn=None
    ):
        pass

    for dataset, model, optimizer, loss_kind in get_current_experiment_flags(
        datasets=[
            Bunch(kind="mnist")
        ],
        models=[
            Bunch(kind='my_model', num_classes=10)
        ],
        optimizers=expand_config(
            base=dict(),
            kind=['adam', 'sgd'],
            lr=[1e-6, 1e-3, 1e-1],
        ),
        loss_names=[
            'softce',
            'softce',
        ]
    ):
        assert dataset.kind == 'mnist'
        assert model.kind == 'my_model'
        assert isinstance(optimizer.kind, str)
        assert optimizer.kind in ['adam', 'sgd']

        setup_training(
            nb_epoch=42,
            dataset=make_dataset(**dataset),
            model=make_model(**model),
            optimizer=make_optimizer(**optimizer),
            loss_fn=make_loss(loss_kind),
        )
