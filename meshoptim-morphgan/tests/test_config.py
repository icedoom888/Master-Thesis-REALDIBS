import pytest
import json
import os
import os.path as op
from pathlib import Path

from noice.utils import Config, make_config, get_variable
from noice.utils.base import mkdir_p


def test_config_constructor():
    """Test construtor."""
    cc = Config(foo='foo', bar='bar')

    assert cc.name == 'default'
    assert cc.foo == 'foo'
    assert cc.bar == 'bar'

    assert cc['name'] == 'default'
    assert cc['foo'] == 'foo'
    assert cc['bar'] == 'bar'


def test_conflicting_constructor_keyword():
    """Test passing `name` twice raises error."""
    with pytest.raises(TypeError, match='multiple values for keyword'):
        _ = Config(name='foo', **{'name': 'bar'})


@pytest.fixture(scope='session')
def dummy_config(tmpdir_factory):
    """Create a dummy confing for testing."""
    tmp_path = tmpdir_factory.mktemp('foo')
    out = tmp_path / 'foo.json'

    config = Config()
    config.num_classes = None
    config.top = 10

    with open(out, 'w') as outfile:
        json.dump(config, outfile)

    return out


def test_config_roundtrip(dummy_config):
    """Test config roundtrip."""
    EXPECTED_CONTENT = {'name': 'default', 'num_classes': None, 'top': 10}
    my_config = make_config(json_file=dummy_config)
    assert len(my_config) == len(EXPECTED_CONTENT)
    for key, expected_val in EXPECTED_CONTENT.items():
        assert my_config[key] == expected_val


@pytest.fixture()
def foo_env(monkeypatch):
    """Create an adequate environment for testing get_variable."""
    monkeypatch.setenv("FOO", "foo")
    assert os.environ["FOO"] == "foo"
    return monkeypatch


@pytest.fixture()
def bar_config(monkeypatch, tmpdir, config_path='.noice-toolbox/config.json'):
    """Create an adequate environment for testing get_variable."""
    assert "BAR" not in os.environ

    monkeypatch.setenv("_FAKE_HOME_DIR", str(tmpdir))
    config = Path(tmpdir) / config_path
    config.parent.mkdir()

    with open(config, 'w') as fid:
        json.dump({'BAR': 'bar'}, fid)

    return monkeypatch


@pytest.fixture()
def broken_config(monkeypatch, tmpdir, config_path='.noice-toolbox/config.json'):
    """Create an adequate environment for testing get_variable."""
    assert "BAR" not in os.environ

    monkeypatch.setenv("_FAKE_HOME_DIR", str(tmpdir))
    config = Path(tmpdir) / config_path
    config.parent.mkdir()

    with open(config, 'w') as fid:
        fid.write("{'BAR': 'bar',}")  # invalid json

    return monkeypatch


def test_get_variable_for_available_elements(foo_env, bar_config):
    """Test available config variables from the environment or project config."""
    assert get_variable(key="FOO") == "foo"
    assert get_variable(key="BAR") == "bar"
    assert get_variable(key="BAR", fallback_value="foobar") == "bar"


def test_get_variable_fallback_behavior(monkeypatch, bar_config, tmpdir):
    """Test configuration variables from the environment or project config."""
    assert get_variable(key="BAR", fallback_value="foobar") == "bar"
    monkeypatch.setenv("_FAKE_HOME_DIR", str(mkdir_p(op.join(tmpdir, 'not_here'))))
    assert get_variable(key="BAR", fallback_value="foobar") == "foobar"


def test_get_variable_errors(tmpdir):
    """Test get_variable common error handling."""
    with pytest.raises(TypeError, match="got .*NotSet"):
        get_variable()

    with pytest.raises(KeyError, match="Key \"BAR\" not found"):
        get_variable(key="BAR", home_dir=Path(tmpdir))


def test_get_variable_broken_config(broken_config):
    """Test get_variable common error handling."""
    with pytest.raises(json.decoder.JSONDecodeError):
        get_variable(key="BAR")
