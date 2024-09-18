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
        },
        "uid": "descriptor-uid",
    }


@pytest.fixture
def stream_resource_factory():
    return lambda chunk_shape: {
        "data_key": "test_img",
        "mimetype": "application/x-hdf5",
        "uri": "file://localhost/test/file/path",
        "resource_path": "test_file.h5",
        "parameters": {
            "path": "entry/data/test_img",
            "swmr": True,
            "chunk_shape": chunk_shape,
        },
        "uid": "stream-resource-uid",
    }


@pytest.fixture
def stream_resource(stream_resource_factory):
    return stream_resource_factory(chunk_shape=None)


@pytest.fixture
def stream_datum_factory():
    return lambda i: {
        "seq_nums": {"start": i + 1, "stop": i + 2},
        "indices": {"start": i, "stop": i + 1},
        "descriptor": "descriptor-uid",
        "stream_resource": "stream-resource-uid",
        "uid": f"stream-datum-uid/{i}",
    }


def test_consolidator_shape(descriptor, stream_resource, stream_datum_factory):
    cons = HDF5Consolidator(stream_resource, descriptor)
    assert cons.shape == (0, 10, 15)
    for i in range(5):
        doc = stream_datum_factory(i)
        cons.consume_stream_datum(doc)
    assert cons.shape == (5, 10, 15)


chunk_shape_testdata = [
    ((), ((5,), (10,), (15,))),
    ((1, 10, 15), ((1, 1, 1, 1, 1), (10,), (15,))),
    ((2,), ((2, 2, 1), (10,), (15,))),
    ((5, 10, 15), ((5,), (10,), (15,))),
    ((10, 10, 15), ((5,), (10,), (15,))),
]


@pytest.mark.parametrize("chunk_shape, expected", chunk_shape_testdata)
def test_consolidator_chunks(descriptor, stream_resource_factory, stream_datum_factory, chunk_shape, expected):
    stream_resource = stream_resource_factory(chunk_shape=chunk_shape)
    cons = HDF5Consolidator(stream_resource, descriptor)
    assert cons.chunks == ((0,), (10,), (15,))
    for i in range(5):
        doc = stream_datum_factory(i)
        cons.consume_stream_datum(doc)
    assert cons.chunks == expected
    assert True
