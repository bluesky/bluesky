import warnings
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from ddt import data, ddt, unpack
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
    absolute_limit_factory,
    auto_redraw,
    fullrange_limit_factory,
    percentile_limit_factory,
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


@auto_redraw
def some_function(obj, x, y):
    return x + y


class TestAutoRedrawDecorator(TestCase):
    def test_function_call_and_return_value(self):
        result = some_function(MagicMock(), 2, 3)
        self.assertEqual(result, 5, "The decorated function should return the sum of inputs")

    def test_redraw_called_when_auto_redraw_true(self):
        mock_obj = MagicMock()
        mock_obj._auto_redraw = True
        some_function(mock_obj, 2, 3)
        mock_obj._fig.canvas.assert_not_called()
        mock_obj._update_artists.assert_called_once()
        mock_obj._draw.assert_called_once()

    # todo complete
    def test_redraw_not_called_when_auto_redraw_false(self):
        self.obj._auto_redraw = False
        self.obj.some_function(2, 3, force_redraw=False)
        self.obj._update_artists.assert_not_called()
        self.obj._draw.assert_not_called()

    # todo complete
    def test_redraw_called_with_force_redraw_true(self):
        self.obj._auto_redraw = False  # Make sure _auto_redraw is false
        self.obj.some_function(2, 3, force_redraw=True)
        self.obj._update_artists.assert_called_once()
        self.obj._draw.assert_called_once()

    # todo complete
    def test_no_canvas_no_redraw(self):
        # Simulate the absence of a canvas
        self.obj._fig.canvas = None
        self.obj.some_function(2, 3)
        self.obj._update_artists.assert_not_called()
        self.obj._draw.assert_not_called()


class TestCrossSection2(TestCase):
    def setUp(self):
        patch_divider = MagicMock()
        patch_divider.append_axes.return_value.plot.return_value = (0,)  # Patch for _ln_v and _ln_h
        patch("bluesky.callbacks._mpl_image_cross_section.make_axes_locatable", return_value=patch_divider).start()
        self.figure = MagicMock()
        self.cross_section = CrossSection(figure=self.figure, colormap="gray", auto_redraw=True)
        self.cross_section._imdata = MagicMock(shape=(100, 100))  # Dummy image data
        self.cross_section._cursor_position_cbs.append(MagicMock())  # Add mock callback

    def test_move_cb_outside_axes(self):
        event = MagicMock()
        event.inaxes = None
        self.cross_section._move_cb(event)
        self.cross_section._cursor_position_cbs[0].assert_not_called()

    # todo complete
    def test_move_cb_valid_event(self):
        event = MagicMock()
        event.inaxes = self.cross_section._image_axes
        event.xdata = 50
        event.ydata = 50
        self.cross_section._move_cb(event)
        self.assertTrue(self.cross_section._ln_h.get_visible())
        self.assertTrue(self.cross_section._ln_v.get_visible())
        self.cross_section._cursor_position_cbs[0].assert_called_once_with(50, 50)

    # todo complete
    def test_click_cb_ignore_outside_click(self):
        event = MagicMock()
        event.inaxes = None
        self.cross_section._click_cb(event)
        # Ensure active state is not toggled
        self.assertTrue(self.cross_section._active)

    # todo complete
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

    # todo complete
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


# todo complete
def test_initialization(live_image):
    assert live_image.field == "test_field"
    # Add more assertions to validate initialization logic


# todo complete
def test_event_handling(live_image):
    mock_doc = {"data": {"test_field": "image_data"}}
    with patch.object(live_image, "update") as mock_update:
        live_image.event(mock_doc)
        mock_update.assert_called_once_with("image_data")


# todo complete
def test_click_callback(mock_cross_section):
    mock_cross_section._click_cb()
    raise AssertionError("Test not implemented")


# todo complete
def test_add_title(mock_cross_section):
    mock_cross_section._figure.canvas.set_window_title("test_title")
    mock_cross_section._figure.show()


# todo complete
def test_connect_callbacks():
    raise AssertionError("Test not implemented")


# todo complete
def test_disconnect_callbacks():
    raise AssertionError("Test not implemented")


# todo complete
def test_artists():
    raise AssertionError("Test not implemented")


# todo complete
def test_active():
    raise AssertionError("Test not implemented")


# todo complete
def test_update_color_map():
    raise AssertionError("Test not implemented")


@ddt
class TestColorLimitFactories(TestCase):
    @data(
        ((0.0, 1.0), np.array([[0.1, 0.5], [0.2, 0.8]]), (0.0, 1.0)),
        ((-10.0, 10.0), np.array([[-5, 5], [-2, 2]]), (-10.0, 10.0)),
    )
    @unpack
    def test_absolute_limits(self, limits, image_data, expected):
        """Test the absolute_limit_factory with different limit arguments."""
        absolute_limit = absolute_limit_factory(limits)
        self.assertEqual(
            absolute_limit(image_data), expected, "Absolute limits should match the provided arguments."
        )

    @data(
        ((0, 100), np.array([[10, 20], [30, 40]]), (10, 40)),
        ((25, 75), np.array([[10, 20], [30, 40]]), (17.5, 32.5)),
        ((0, 100), np.array([[np.nan, np.nan], [np.nan, np.nan]]), (np.nan, np.nan)),
    )
    @unpack
    def test_percentile_limits(self, percentiles, image_data, expected):
        """Test the percentile_limit_factory with different percentile arguments."""
        percentile_limit = percentile_limit_factory(percentiles)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = percentile_limit(image_data)
            if all(np.isnan(x) for x in expected):
                self.assertTrue(all(np.isnan(x) for x in result), "Should return (nan, nan) for all NaN values")
                self.assertEqual(len(w), 0, "No warnings should be issued for nanpercentile with all NaN data")
            else:
                self.assertEqual(
                    result, expected, "Percentile limits should be correctly calculated based on the data"
                )

    @data(
        (np.array([[1, 2], [3, 4]]), (1, 4)),
        (np.array([[np.nan, 2], [3, np.nan]]), (2, 3)),
        (np.array([[np.nan, np.nan], [np.nan, np.nan]]), (np.nan, np.nan)),
    )
    @unpack
    def test_full_range_limits(self, image_data, expected):
        """Test the fullrange_limit_factory for different data scenarios."""
        full_range = fullrange_limit_factory()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Trigger all warnings to be caught
            result = full_range(image_data)
            # Check if warning is raised correctly for all NaN values
            if np.isnan(expected[0]) and np.isnan(expected[1]):
                self.assertEqual(len(w), 2)  # Ensure one warning was raised
                self.assertTrue(issubclass(w[-1].category, RuntimeWarning))  # Check warning type
                self.assertTrue("All-NaN slice encountered" in str(w[-1].message))  # Check warning message
                self.assertTrue(
                    np.isnan(result[0]) and np.isnan(result[1]), "Should return (nan, nan) for all NaN values"
                )
            else:
                self.assertEqual(result, expected, "Full range should correctly calculate min and max of the data")
