from math import ceil

import pytest

from bluesky.consolidators import HDF5Consolidator, consolidator_factory


@pytest.fixture
def descriptor():
    return {
        "data_keys": {
            "test_img": {
                "shape": [10, 15],
                "dtype": "array",
                "dtype_numpy": "<f8",
                "external": "STREAM:",
                "object_name": "test_object",
            },
            "test_cube": {
                "shape": [10, 15, 3],
                "dtype": "array",
                "dtype_numpy": "<f8",
                "external": "STREAM:",
                "object_name": "test_object",
            },
            "test_arr": {
                "shape": [1],
                "dtype": "array",
                "dtype_numpy": "<f8",
                "external": "STREAM:",
                "object_name": "test_object",
            },
            "test_num": {
                "shape": [],
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


shape_testdata = [
    # ("test_img", True, (5, 10, 15)),
    # ("test_cube", True, (5, 10, 15, 3)),
    # ("test_arr", True, (5,)),
    # ("test_num", True, (5,)),
    ("test_img", False, (50, 15)),
    ("test_cube", False, (50, 15, 3)),
    ("test_arr", False, (5,)),
    ("test_num", False, (5,)),
]


@pytest.mark.parametrize("data_key, stackable, expected", shape_testdata)
def test_hdf5_shape(descriptor, hdf5_stream_resource_factory, stream_datum_factory, data_key, stackable, expected):
    stream_resource = hdf5_stream_resource_factory(data_key=data_key, chunk_shape=())
    cons = HDF5Consolidator(stream_resource, descriptor)
    cons.stackable = stackable
    assert cons.shape == (0, *expected[1:])
    for i in range(5):
        doc = stream_datum_factory(data_key, i, i, i + 1)
        cons.consume_stream_datum(doc)
    assert cons.shape == expected


supported_image_seq_formats = ["jpeg", "tiff"]


@pytest.mark.parametrize("data_key, stackable, expected", shape_testdata)
@pytest.mark.parametrize("image_format", supported_image_seq_formats)
@pytest.mark.parametrize("files_per_stream_datum", [1, 2, 3, 5])
def test_tiff_shape(
    descriptor,
    image_seq_stream_resource_factory,
    stream_datum_factory,
    image_format,
    data_key,
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
    assert len(cons.assets) == 5


chunk_testdata = [
    ("test_img", True, False, (), ((5,), (10,), (15,))),
    ("test_img", True, False, (1, 10, 15), ((1, 1, 1, 1, 1), (10,), (15,))),
    ("test_img", True, False, (2,), ((2, 2, 1), (10,), (15,))),
    ("test_img", True, False, (5, 10, 15), ((5,), (10,), (15,))),
    ("test_img", True, False, (10, 10, 15), ((5,), (10,), (15,))),
    ("test_img", True, False, (3, 4, 5), ((3, 2), (4, 4, 2), (5, 5, 5))),
    ("test_cube", True, False, (3, 4, 5, 3), ((3, 2), (4, 4, 2), (5, 5, 5), (3,))),
    ("test_arr", True, False, (), ((5,),)),
    ("test_arr", True, False, (2,), ((2, 2, 1),)),
    ("test_num", True, False, (), ((5,),)),
    ("test_num", True, False, (2,), ((2, 2, 1),)),
    ("test_img", False, False, (), ((50,), (15,))),
    ("test_img", False, False, (10, 15), ((10, 10, 10, 10, 10), (15,))),
    ("test_img", False, False, (7,), ((7, 3, 7, 3, 7, 3, 7, 3, 7, 3), (15,))),
    ("test_img", True, True, (), ((5,), (10,), (15,))),
    ("test_img", True, True, (1, 10, 15), ((1, 1, 1, 1, 1), (10,), (15,))),
    ("test_img", True, True, (2,), ((2, 2, 1), (10,), (15,))),
    ("test_img", True, True, (5, 10, 15), ((5,), (10,), (15,))),
    ("test_img", True, True, (10, 10, 15), ((5,), (10,), (15,))),
    ("test_img", True, True, (3, 4, 5), ((3, 2), (4, 4, 2), (5, 5, 5))),
    ("test_cube", True, True, (3, 4, 5, 3), ((3, 2), (4, 4, 2), (5, 5, 5), (3,))),
    ("test_arr", True, True, (), ((5,),)),
    ("test_arr", True, True, (2,), ((2, 2, 1),)),
    ("test_num", True, True, (), ((5,),)),
    ("test_num", True, True, (2,), ((2, 2, 1),)),
    ("test_img", False, True, (), ((50,), (15,))),
    ("test_img", False, True, (10, 15), ((10, 10, 10, 10, 10), (15,))),
    ("test_img", False, True, (7,), ((7, 7, 7, 7, 7, 7, 7, 1), (15,))),
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
