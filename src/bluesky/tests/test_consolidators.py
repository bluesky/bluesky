from math import ceil

import pytest

from bluesky.consolidators import HDF5Consolidator, consolidator_factory


@pytest.fixture
def descriptor():
    return {
        "data_keys": {
            "test_img": {
                "shape": [1, 10, 15],
                "dtype": "array",
                "dtype_numpy": "<f8",
                "external": "STREAM:",
                "object_name": "test_object",
            },
            "test_7_imgs": {
                "shape": [7, 10, 15],
                "dtype": "array",
                "dtype_numpy": "<f8",
                "external": "STREAM:",
                "object_name": "test_object",
            },
            "test_cube": {
                "shape": [1, 10, 15, 3],
                "dtype": "array",
                "dtype_numpy": "<f8",
                "external": "STREAM:",
                "object_name": "test_object",
            },
            "test_7_cubes": {
                "shape": [7, 10, 15, 3],
                "dtype": "array",
                "dtype_numpy": "<f8",
                "external": "STREAM:",
                "object_name": "test_object",
            },
            "test_arr": {
                "shape": [
                    1,
                    3,
                ],
                "dtype": "array",
                "dtype_numpy": "<f8",
                "external": "STREAM:",
                "object_name": "test_object",
            },
            "test_7_arrs": {
                "shape": [
                    7,
                    3,
                ],
                "dtype": "array",
                "dtype_numpy": "<f8",
                "external": "STREAM:",
                "object_name": "test_object",
            },
            "test_num": {
                "shape": [
                    1,
                ],
                "dtype": "number",
                "dtype_numpy": "<f8",
                "external": "STREAM:",
                "object_name": "test_object",
            },
            "test_7_nums": {
                "shape": [
                    7,
                ],
                "dtype": "number",
                "dtype_numpy": "<f8",
                "external": "STREAM:",
                "object_name": "test_object",
            },
        },
        "uid": "descriptor-uid",
    }


@pytest.fixture
def hdf5_stream_resource_factory():
    return lambda data_key, chunk_shape: {
        "data_key": data_key,
        "mimetype": "application/x-hdf5",
        "uri": "file://localhost/test/file/path",
        "resource_path": "test_file.h5",
        "parameters": {
            "dataset": f"entry/data/{data_key}",
            "swmr": True,
            "chunk_shape": chunk_shape,
        },
        "uid": f"stream-resource-uid-{data_key}",
    }


@pytest.fixture
def image_seq_stream_resource_factory():
    return lambda image_format, data_key, chunk_shape: {
        "data_key": data_key,
        "mimetype": f"multipart/related;type=image/{image_format}",
        "uri": "file://localhost/test/file/path",
        "parameters": {"chunk_shape": chunk_shape, "template": "img_{:06d}." + image_format},
        "uid": f"stream-resource-uid-{data_key}",
    }


@pytest.fixture
def stream_datum_factory():
    return lambda data_key, indx, i_start, i_stop: {
        "seq_nums": {"start": i_start + 1, "stop": i_stop + 1},
        "indices": {"start": i_start, "stop": i_stop},
        "descriptor": "descriptor-uid",
        "stream_resource": f"stream-resource-uid-{data_key}",
        "uid": f"stream-datum-uid-{data_key}/{indx}",
    }


# Tuples of (data_key, points_per_event, stackable, expected_shape)
shape_testdata = [
    # 5 events, 1 or 7 image per event, 10x15 pixels
    ("test_img", 1, False, (5, 10, 15)),
    ("test_7_imgs", 7, False, (35, 10, 15)),
    ("test_img", 1, True, (5, 1, 10, 15)),
    ("test_7_imgs", 7, True, (5, 7, 10, 15)),
    # 5 events, 1 or 7 cube per event, 10x15x3 pixels
    ("test_cube", 1, False, (5, 10, 15, 3)),
    ("test_7_cubes", 7, False, (35, 10, 15, 3)),
    ("test_cube", 1, True, (5, 1, 10, 15, 3)),
    ("test_7_cubes", 7, True, (5, 7, 10, 15, 3)),
    # 5 events, 1 or 7 array per event, 1 element in array
    ("test_arr", 1, False, (5, 3)),
    ("test_7_arrs", 7, False, (35, 3)),
    ("test_arr", 1, True, (5, 1, 3)),
    ("test_7_arrs", 7, True, (5, 7, 3)),
    # 5 events, 1 or 7 number per event
    ("test_num", 1, False, (5,)),
    ("test_7_nums", 7, False, (35,)),
    ("test_num", 1, True, (5, 1)),
    ("test_7_nums", 7, True, (5, 7)),
]


@pytest.mark.parametrize("data_key, points_per_event, stackable, expected", shape_testdata)
def test_hdf5_shape(
    descriptor, hdf5_stream_resource_factory, stream_datum_factory, data_key, points_per_event, stackable, expected
):
    stream_resource = hdf5_stream_resource_factory(data_key=data_key, chunk_shape=())
    cons = HDF5Consolidator(stream_resource, descriptor)
    cons.stackable = stackable
    assert cons.shape == (0, *expected[1:])
    for i in range(5):
        doc = stream_datum_factory(data_key, i, i, i + 1)
        cons.consume_stream_datum(doc)
    assert cons.shape == expected


supported_image_seq_formats = ["jpeg", "tiff"]


@pytest.mark.parametrize("data_key, points_per_event, stackable, expected", shape_testdata)
@pytest.mark.parametrize("image_format", supported_image_seq_formats)
@pytest.mark.parametrize("files_per_stream_datum", [1, 2, 3, 5])
def test_tiff_and_jpeg_shape(
    descriptor,
    image_seq_stream_resource_factory,
    stream_datum_factory,
    image_format,
    data_key,
    points_per_event,
    stackable,
    expected,
    files_per_stream_datum,
):
    stream_resource = image_seq_stream_resource_factory(
        image_format=image_format, data_key=data_key, chunk_shape=(1,)
    )
    cons = consolidator_factory(stream_resource, descriptor)
    cons.stackable = stackable
    assert cons.shape == (0, *expected[1:])
    for i in range(ceil(5 / files_per_stream_datum)):
        doc = stream_datum_factory(
            data_key, i, i * files_per_stream_datum, min((i + 1) * files_per_stream_datum, 5)
        )
        cons.consume_stream_datum(doc)
    assert cons.shape == expected

    # Stackable case here corresponds to multipage tiffs (AD does not support them though)
    assert len(cons.assets) == 5 * points_per_event if not stackable else 5


# Tuples of (data_key, stackable, join_chunks, chunk_shape, expected_chunks)
chunk_testdata = [
    ("test_img", True, True, (), ((5,), (1,), (10,), (15,))),
    ("test_img", True, True, (1, 1, 10, 15), ((1, 1, 1, 1, 1), (1,), (10,), (15,))),
    ("test_img", True, True, (2,), ((2, 2, 1), (1,), (10,), (15,))),
    ("test_img", True, True, (5, 1, 10, 15), ((5,), (1,), (10,), (15,))),
    ("test_img", True, True, (10, 1), ((5,), (1,), (10,), (15,))),
    ("test_img", True, True, (3, 1, 4, 5), ((3, 2), (1,), (4, 4, 2), (5, 5, 5))),
    ("test_7_imgs", True, True, (), ((5,), (7,), (10,), (15,))),
    ("test_7_imgs", True, True, (1, 1), ((1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1), (10,), (15,))),
    ("test_7_imgs", True, True, (2,), ((2, 2, 1), (7,), (10,), (15,))),
    ("test_7_imgs", True, True, (5, 1, 10, 15), ((5,), (1, 1, 1, 1, 1, 1, 1), (10,), (15,))),
    ("test_7_imgs", True, True, (10, 5), ((5,), (5, 2), (10,), (15,))),
    (
        "test_7_imgs",
        True,
        True,
        (3, 4, 5, 6),
        (
            (3, 2),
            (4, 3),
            (5, 5),
            (6, 6, 3),
        ),
    ),
    ("test_cube", True, True, (3, 1, 4, 5, 3), ((3, 2), (1,), (4, 4, 2), (5, 5, 5), (3,))),
    ("test_7_cubes", True, True, (3, 4, 5, 6, 7), ((3, 2), (4, 3), (5, 5), (6, 6, 3), (3,))),
    ("test_arr", True, True, (5, 1, 1), ((5,), (1,), (1, 1, 1))),
    ("test_arr", True, True, (2,), ((2, 2, 1), (1,), (3,))),
    ("test_7_arrs", True, True, (5, 1, 1), ((5,), (1, 1, 1, 1, 1, 1, 1), (1, 1, 1))),
    ("test_7_arrs", True, True, (2,), ((2, 2, 1), (7,), (3,))),
    ("test_num", True, True, (), ((5,), (1,))),
    ("test_num", True, True, (2,), ((2, 2, 1), (1,))),
    ("test_7_nums", True, True, (), ((5,), (7,))),
    ("test_7_nums", True, True, (2,), ((2, 2, 1), (7,))),
    ("test_7_nums", True, True, (2, 3), ((2, 2, 1), (3, 3, 1))),
    ("test_img", False, True, (), ((5,), (10,), (15,))),
    ("test_img", False, True, (1, 10, 15), ((1, 1, 1, 1, 1), (10,), (15,))),
    ("test_img", False, True, (2,), ((2, 2, 1), (10,), (15,))),
    ("test_img", False, True, (5, 10, 15), ((5,), (10,), (15,))),
    ("test_img", False, True, (10, 1), ((5,), (1,) * 10, (15,))),
    ("test_img", False, True, (3, 4, 5), ((3, 2), (4, 4, 2), (5, 5, 5))),
    ("test_7_imgs", False, True, (), ((35,), (10,), (15,))),
    ("test_7_imgs", False, False, (), ((35,), (10,), (15,))),
    ("test_7_imgs", False, True, (1, 1), ((1,) * 35, (1,) * 10, (15,))),
    ("test_7_imgs", False, False, (2,), ((2, 2, 2, 1) * 5, (10,), (15,))),
    ("test_7_imgs", False, True, (2,), ((2,) * 17 + (1,), (10,), (15,))),
    ("test_7_imgs", False, True, (5, 10, 15), ((5,) * 7, (10,), (15,))),
    ("test_7_imgs", False, False, (5, 10, 15), ((5, 2) * 5, (10,), (15,))),
    ("test_7_imgs", False, True, (10, 5), ((10, 10, 10, 5), (5, 5), (15,))),
    ("test_7_imgs", False, False, (10, 5), ((7,) * 5, (5, 5), (15,))),
    (
        "test_7_imgs",
        False,
        True,
        (3, 5, 6),
        (
            (3,) * 11 + (2,),
            (5, 5),
            (6, 6, 3),
        ),
    ),
    (
        "test_7_imgs",
        False,
        False,
        (3, 5, 6),
        (
            (3, 3, 1) * 5,
            (5, 5),
            (6, 6, 3),
        ),
    ),
    ("test_cube", False, True, (3, 4, 5, 3), ((3, 2), (4, 4, 2), (5, 5, 5), (3,))),
    ("test_7_cubes", False, True, (3, 5, 6, 7), ((3,) * 11 + (2,), (5, 5), (6, 6, 3), (3,))),
    ("test_7_cubes", False, False, (3, 5, 6, 7), ((3, 3, 1) * 5, (5, 5), (6, 6, 3), (3,))),
    ("test_arr", False, True, (5, 1), ((5,), (1, 1, 1))),
    ("test_arr", False, True, (2,), ((2, 2, 1), (3,))),
    ("test_7_arrs", False, True, (5, 1), ((5,) * 7, (1, 1, 1))),
    ("test_7_arrs", False, False, (5, 1), ((5, 2) * 5, (1, 1, 1))),
    ("test_7_arrs", False, True, (2,), ((2,) * 17 + (1,), (3,))),
    ("test_7_arrs", False, False, (2,), ((2, 2, 2, 1) * 5, (3,))),
    ("test_num", False, True, (), ((5,),)),
    ("test_num", False, True, (2,), ((2, 2, 1),)),
    ("test_7_nums", False, True, (), ((35,),)),
    ("test_7_nums", False, False, (), ((35,),)),
    ("test_7_nums", False, True, (3,), ((3,) * 11 + (2,),)),
    ("test_7_nums", False, False, (3,), ((3, 3, 1) * 5,)),
]


@pytest.mark.parametrize("data_key, stackable, join_chunks, chunk_shape, expected", chunk_testdata)
def test_chunks(
    descriptor,
    hdf5_stream_resource_factory,
    stream_datum_factory,
    data_key,
    stackable,
    join_chunks,
    chunk_shape,
    expected,
):
    stream_resource = hdf5_stream_resource_factory(data_key=data_key, chunk_shape=chunk_shape)
    cons = HDF5Consolidator(stream_resource, descriptor)
    cons.stackable = stackable
    cons.join_chunks = join_chunks
    assert cons.chunks == ((0,), *expected[1:])
    for i in range(5):
        doc = stream_datum_factory(data_key, i, i, i + 1)
        cons.consume_stream_datum(doc)
    assert cons.chunks == expected
