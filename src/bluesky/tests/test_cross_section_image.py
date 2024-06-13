from unittest.mock import patch

import pytest
from matplotlib import colormaps
from matplotlib.colors import (
    Colormap,
    Normalize,
)

from bluesky.callbacks._mpl_image_cross_section import (
    InterpolationEnum,
)
from bluesky.callbacks.broker import LiveImage


@pytest.fixture
def mock_matplotlib():
    with patch("matplotlib.pyplot.figure") as mock_fig:
        yield mock_fig


@pytest.fixture
def live_image(mock_matplotlib):
    return LiveImage(
        "test_field",
        cmap=Colormap(colormaps["magma"]),
        norm=Normalize(),
        interpolation=InterpolationEnum.NONE,
    )


def test_initialization(live_image):
    c = colormaps
    assert live_image.field == "test_field"
    # Add more assertions to validate initialization logic


def test_event_handling(live_image):
    mock_doc = {"data": {"test_field": "image_data"}}
    with patch.object(live_image, "update") as mock_update:
        live_image.event(mock_doc)
        mock_update.assert_called_once_with("image_data")


def test_auto_redraw():
    raise AssertionError("Test not implemented")


def test_cross_section_init():
    raise AssertionError("Test not implemented")


def test_move_callback():
    raise AssertionError("Test not implemented")


def test_click_callback():
    raise AssertionError("Test not implemented")


def test_connect_callbacks():
    raise AssertionError("Test not implemented")


def test_disconnect_callbacks():
    raise AssertionError("Test not implemented")


def test_artists():
    # todo init
    # todo update
    raise AssertionError("Test not implemented")


def test_active():
    raise AssertionError("Test not implemented")


def test_update_color_map():
    raise AssertionError("Test not implemented")


def test_disconnect_callbacks():
    raise AssertionError("Test not implemented")


def test_fullrange_limit_factory():
    raise AssertionError("Test not implemented")


def test_absolute_limit_factory():
    raise AssertionError("Test not implemented")


def test_percentile_limit_factory():
    raise AssertionError("Test not implemented")
