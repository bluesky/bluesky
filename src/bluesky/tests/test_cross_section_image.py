from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.colors import (
    Colormap,
    Normalize,
)
from matplotlib.figure import Figure

from bluesky.callbacks._mpl_image_cross_section import (
    CrossSection,
    InterpolationEnum,
    auto_redraw,
)
from bluesky.callbacks.broker import LiveImage


@pytest.fixture
def mock_matplotlib():
    with patch("matplotlib.pyplot.figure") as mock_fig:
        mock_figure_instance = MagicMock(spec=plt.Figure)
        mock_subplot = MagicMock(spec=plt.Axes)
        mock_figure_instance.add_subplot.return_value = mock_subplot
        mock_subplot.get_figure.return_value = mock_figure_instance
        mock_fig.return_value = mock_figure_instance
        yield mock_fig


@pytest.fixture
def mock_cross_section():
    return CrossSection(figure=MagicMock())


@pytest.fixture
def live_image(mock_cross_section):
    return LiveImage(
        "test_field",
        cmap=Colormap(colormaps["magma"]),
        norm=Normalize(),
        interpolation=InterpolationEnum.NONE,
        cross_section=mock_cross_section,
    )


class TestAutoRedrawDecorator(TestCase):
    def setUp(self):
        class TestClass:
            def __init__(self):
                self._fig = MagicMock()
                self._auto_redraw = True

            @auto_redraw
            def some_function(self, x, y):
                return x + y

            def _update_artists(self):
                pass

            def _draw(self):
                pass

        self.obj = TestClass()

    def test_function_call_and_return_value(self):
        result = self.obj.some_function(2, 3)
        self.assertEqual(result, 5, "The decorated function should return the sum of inputs")

    def test_redraw_called_when_auto_redraw_true(self, mock_cross_section: CrossSection):
        mock_cross_section.draw_idle()
        self.obj.some_function(2, 3)
        self.obj._fig.canvas.assert_not_called()
        self.obj._update_artists.assert_called_once()
        self.obj._draw.assert_called_once()

    def test_redraw_not_called_when_auto_redraw_false(self):
        self.obj._auto_redraw = False
        self.obj.some_function(2, 3, force_redraw=False)
        self.obj._update_artists.assert_not_called()
        self.obj._draw.assert_not_called()

    def test_redraw_called_with_force_redraw_true(self):
        self.obj._auto_redraw = False  # Make sure _auto_redraw is false
        self.obj.some_function(2, 3, force_redraw=True)
        self.obj._update_artists.assert_called_once()
        self.obj._draw.assert_called_once()

    def test_no_canvas_no_redraw(self):
        # Simulate the absence of a canvas
        self.obj._fig.canvas = None
        self.obj.some_function(2, 3)
        self.obj._update_artists.assert_not_called()
        self.obj._draw.assert_not_called()


class TestCrossSection(TestCase):
    def setUp(self):
        self.figure = MagicMock(spec=Figure)
        self.cross_section = CrossSection(figure=self.figure, colormap="gray", auto_redraw=True)
        self.cross_section._imdata = MagicMock(shape=(100, 100))  # Dummy image data
        self.cross_section._cursor_position_cbs.append(MagicMock())  # Add mock callback

    @patch("mpl_toolkits.axes_grid1.make_axes_locatable")
    def test_move_cb_outside_axes(self):
        event = MagicMock()
        event.inaxes = None
        self.cross_section._move_cb(event)
        self.cross_section._cursor_position_cbs[0].assert_not_called()

    def test_move_cb_valid_event(self):
        event = MagicMock()
        event.inaxes = self.cross_section._image_axes
        event.xdata = 50
        event.ydata = 50
        self.cross_section._move_cb(event)
        self.assertTrue(self.cross_section._ln_h.get_visible())
        self.assertTrue(self.cross_section._ln_v.get_visible())
        self.cross_section._cursor_position_cbs[0].assert_called_once_with(50, 50)

    def test_click_cb_ignore_outside_click(self):
        event = MagicMock()
        event.inaxes = None
        self.cross_section._click_cb(event)
        # Ensure active state is not toggled
        self.assertTrue(self.cross_section._active)

    def test_click_cb_toggle_active(self):
        event = MagicMock()
        event.inaxes = self.cross_section._image_axes
        initial_state = self.cross_section._active
        self.cross_section._click_cb(event)
        self.assertNotEqual(self.cross_section._active, initial_state)


class TestInitArtists(TestCase):
    def setUp(self):
        # Mocking necessary matplotlib components
        self.mock_figure = MagicMock()
        self.mock_im = MagicMock()
        self.mock_axes = MagicMock()
        self.mock_ln_v = MagicMock()
        self.mock_ln_h = MagicMock()
        self.mock_canvas = MagicMock()

        # Instantiate CrossSection with mocked figure
        self.cross_section = CrossSection(figure=self.mock_figure)
        self.cross_section._im = self.mock_im
        self.cross_section._image_axes = self.mock_axes
        self.cross_section._ln_v = self.mock_ln_v
        self.cross_section._ln_h = self.mock_ln_h
        self.cross_section._figure.canvas = self.mock_canvas

    def test_init_artists(self):
        # Create a dummy image array
        dummy_image = np.random.rand(10, 15)

        # Call _init_artists with the dummy image
        self.cross_section._init_artists(dummy_image)

        # Check if image data is set correctly
        self.assertTrue(np.array_equal(self.cross_section._imdata, dummy_image))

        # Check if the image extent and axes limits are set correctly
        self.mock_im.set_extent.assert_called_once_with([-0.5, 15.5, 10.5, -0.5])
        self.mock_axes.set_xlim.assert_called_once_with([-0.05, 15.5])
        self.mock_axes.set_ylim.assert_called_once_with([10.5, -0.5])

        # Check if the format_coord function was replaced
        self.assertIsNotNone(self.cross_section._image_axes.format_coord)
        # Example to check format_coord output
        coord_output = self.cross_section._image_axes.format_coord(5, 5)
        self.assertIn("X: 5 Y: 5 I:", coord_output)

        # Check vertical and horizontal line data setup
        self.mock_ln_v.set_data.assert_called_once_with(np.zeros(10), np.arange(10))
        self.mock_ln_h.set_data.assert_called_once_with(np.arange(15), np.zeros(15))

        # Check if callbacks were connected
        self.mock_canvas.assert_has_calls([call.mpl_connect()])

        # Check if the dirty flag was set
        self.assertTrue(self.cross_section._dirty)


def test_initialization(live_image):
    assert live_image.field == "test_field"
    # Add more assertions to validate initialization logic


def test_event_handling(live_image):
    mock_doc = {"data": {"test_field": "image_data"}}
    with patch.object(live_image, "update") as mock_update:
        live_image.event(mock_doc)
        mock_update.assert_called_once_with("image_data")


def test_auto_redraw():
    raise AssertionError("Test not implemented")


def test_cross_section_init(mock_cross_section):
    print(mock_cross_section)
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


def test_fullrange_limit_factory():
    raise AssertionError("Test not implemented")


def test_absolute_limit_factory():
    raise AssertionError("Test not implemented")


def test_percentile_limit_factory():
    raise AssertionError("Test not implemented")
