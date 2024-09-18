import pytest

from bluesky.consolidators import HDF5Consolidator


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
def stream_resource_factory():
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
def stream_datum_factory():
    return lambda data_key, i: {
        "seq_nums": {"start": i + 1, "stop": i + 2},
        "indices": {"start": i, "stop": i + 1},
        "descriptor": "descriptor-uid",
        "stream_resource": f"stream-resource-uid-{data_key}",
        "uid": f"stream-datum-uid-{data_key}/{i}",
    }


shape_testdata = [
    ("test_img", (5, 10, 15)),
    ("test_cube", (5, 10, 15, 3)),
    ("test_arr", (5,)),
    ("test_num", (5,)),
]


@pytest.mark.parametrize("data_key, expected", shape_testdata)
def test_shape(descriptor, stream_resource_factory, stream_datum_factory, data_key, expected):
    stream_resource = stream_resource_factory(data_key=data_key, chunk_shape=None)
    cons = HDF5Consolidator(stream_resource, descriptor)
    assert cons.shape == (0, *expected[1:])
    for i in range(5):
        doc = stream_datum_factory(data_key, i)
        cons.consume_stream_datum(doc)
    assert cons.shape == expected


chunk_testdata = [
    ("test_img", (), ((5,), (10,), (15,))),
    ("test_img", (1, 10, 15), ((1, 1, 1, 1, 1), (10,), (15,))),
    ("test_img", (2,), ((2, 2, 1), (10,), (15,))),
    ("test_img", (5, 10, 15), ((5,), (10,), (15,))),
    ("test_img", (10, 10, 15), ((5,), (10,), (15,))),
    ("test_img", (3, 4, 5), ((3, 2), (4, 4, 2), (5, 5, 5))),
    ("test_cube", (3, 4, 5, 3), ((3, 2), (4, 4, 2), (5, 5, 5), (3,))),
    ("test_arr", (), ((5,),)),
    ("test_arr", (2,), ((2, 2, 1),)),
    ("test_num", (), ((5,),)),
    ("test_num", (2,), ((2, 2, 1),)),
]


@pytest.mark.parametrize("data_key, chunk_shape, expected", chunk_testdata)
def test_chunks(descriptor, stream_resource_factory, stream_datum_factory, data_key, chunk_shape, expected):
    stream_resource = stream_resource_factory(data_key=data_key, chunk_shape=chunk_shape)
    cons = HDF5Consolidator(stream_resource, descriptor)
    assert cons.chunks == ((0,), *expected[1:])
    for i in range(5):
        doc = stream_datum_factory(data_key, i)
        cons.consume_stream_datum(doc)
    assert cons.chunks == expected
