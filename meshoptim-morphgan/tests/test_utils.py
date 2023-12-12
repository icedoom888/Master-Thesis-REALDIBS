# -*- coding: utf-8 -*-
# Authors: Joan Massich <joan.massich@disneyresearch.com>
#

import pytest
import pickle
from noice.utils import BunchConstNamed
from noice.utils.base import _check_option
from noice.utils._bunch import NamedInt, NamedFloat


def test_bunch():
    """Test some Bunch functionalities."""
    bb = BunchConstNamed()
    bb.foo = 42
    bb.bar = 3.14
    assert isinstance(bb.foo, int)
    assert isinstance(bb.foo, NamedInt)
    assert repr(bb.foo) == 'foo: 42'
    assert isinstance(bb.bar, float)
    assert isinstance(bb.bar, NamedFloat)
    assert repr(bb.bar) == 'bar: 3.14'

    new_bb = pickle.loads(pickle.dumps(bb))
    assert new_bb == bb


def test_check_options():
    """Test options checker."""
    assert _check_option(item='foo', allowed_values=('foo', 'bar'))
    assert _check_option(item='foo', allowed_values=['foo', 'bar'])
    assert _check_option(
        item='foo',
        allowed_values=dict(foo='', bar='').keys()
    )


@pytest.mark.parametrize(
    'allowed_values, msg',
    [
        pytest.param(
            ('foo',),
            "Invalid value. The only allowed value is 'foo', but got 'oof' instead.",
            id="oof in (foo,)",
        ),
        pytest.param(
            ('foo', 'bar'),
            "Invalid value. Allowed values are 'foo' and 'bar', but got 'oof' instead.",
            id="oof in (foo, bar)",
        ),
        pytest.param(
            ('foo', 'bar', 'baz'),
            ("Invalid value. Allowed values are 'foo', 'bar' and 'baz', but got"
             " 'oof' instead."),
            id="oof in (foo, bar, baz)",
        ),
        pytest.param(
            (),
            "Invalid value. The only allowed value is \(\), but got 'oof' instead.",  # noqa
            id="oof in ()",
        ),
        pytest.param(
            'foo',
            "it does not matter",
            marks=[pytest.mark.xfail],  # TypeError
            id="oof in 'foo' --> TypeError",
        ),
    ]
)
def test_check_options_allowed_values_error_wording(allowed_values, msg):
    """Test error workding for different allowed_values."""
    with pytest.raises(ValueError, match=msg):
        _check_option(item='oof', allowed_values=allowed_values)


@pytest.mark.parametrize(
    'item_name, extra, msg',
    [
        pytest.param('name', '', (
            "Invalid value for name. "
            "Allowed values are 'foo' and 'bar', but got 'oof' instead."),
            id="name",
        ),
        pytest.param('name', 'when condition A', (
            "Invalid value for name when condition A. "
            "Allowed values are 'foo' and 'bar', but got 'oof' instead."),
            id="name + extra",
        ),
    ]
)
def test_check_options_item_name_wording(item_name, extra, msg):
    """Test options checker item name wording."""
    with pytest.raises(ValueError, match=msg):
        _check_option(
            item='oof', allowed_values=('foo', 'bar'), item_name=item_name, extra=extra
        )
