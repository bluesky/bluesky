import pytest
from bluesky.utils import register_transform


@pytest.fixture
def transform_cell():
    IPython = pytest.importorskip('IPython')
    ip = IPython.core.interactiveshell.InteractiveShell()
    register_transform('RE', prefix='<', ip=ip)
    if IPython.__version__ >= '7':
        return ip.transform_cell
    else:
        return ip.input_splitter.transform_cell


def test_register_transform_smoke(transform_cell):
    assert True


@pytest.mark.parametrize('cell',
                         ['a < b\n',
                          'RE(plan(a < b))\n',
                          'for j in range(5):\n    < a\n    <b\n'])
def test_no_transform(transform_cell, cell):
    new_cell = transform_cell(cell)
    assert cell == new_cell


@pytest.mark.parametrize('cell',
                         ['< b\n',
                          '<a\n',
                          '<fly(a, b, c), md={"a": "aardvark"}\n',
                          '  <b\n',
                          '<a, c=d\n'])
def test_transform(transform_cell, cell):
    new_cell = transform_cell(cell)

    assert f'RE({cell.lstrip("< ").strip()})\n' == new_cell
