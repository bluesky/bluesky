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
            "test_6_imgs": {
                "shape": [6, 10, 15],
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
    format_to_mimetype = {
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "tiff": "image/tiff",
        "tif": "image/tiff",
    }
    return lambda image_format, data_key, chunk_shape: {
        "data_key": data_key,
        "mimetype": f"multipart/related;type={format_to_mimetype[image_format]}",
        "uri": "file://localhost/test/file/path",
        "parameters": {"chunk_shape": chunk_shape, "template": "img_{:06d}." + image_format},
        "uid": f"stream-resource-uid-{data_key}",
    }


@pytest.fixture
def csv_stream_resource_factory():
    return lambda data_key, chunk_shape: {
        "data_key": data_key,
        "mimetype": "text/csv;header=absent",
        "uri": "file://localhost/test/file/path",
        "resource_path": "test_file.csv",
        "parameters": {"chunk_shape": chunk_shape},
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


# Tuples of (data_key, frames_per_datum, join_method, expected_shape)
shape_testdata = [
    # 5 events, 1 or 7 image per event, 10x15 pixels
    ("test_img", 1, "concat", (5, 10, 15)),
    ("test_7_imgs", 7, "concat", (35, 10, 15)),
    ("test_img", 1, "stack", (5, 1, 10, 15)),
    ("test_7_imgs", 7, "stack", (5, 7, 10, 15)),
    # 5 events, 1 or 7 cube per event, 10x15x3 pixels
    ("test_cube", 1, "concat", (5, 10, 15, 3)),
    ("test_7_cubes", 7, "concat", (35, 10, 15, 3)),
    ("test_cube", 1, "stack", (5, 1, 10, 15, 3)),
    ("test_7_cubes", 7, "stack", (5, 7, 10, 15, 3)),
    # 5 events, 1 or 7 array per event, 1 element in array
    ("test_arr", 1, "concat", (5, 3)),
    ("test_7_arrs", 7, "concat", (35, 3)),
    ("test_arr", 1, "stack", (5, 1, 3)),
    ("test_7_arrs", 7, "stack", (5, 7, 3)),
    # 5 events, 1 or 7 number per event
    ("test_num", 1, "concat", (5,)),
    ("test_7_nums", 7, "concat", (35,)),
    ("test_num", 1, "stack", (5, 1)),
    ("test_7_nums", 7, "stack", (5, 7)),
]


@pytest.mark.parametrize("data_key, frames_per_datum, join_method, expected", shape_testdata)
def test_hdf5_shape(
    descriptor,
    hdf5_stream_resource_factory,
    stream_datum_factory,
    data_key,
    frames_per_datum,
    join_method,
    expected,
):
    stream_resource = hdf5_stream_resource_factory(data_key=data_key, chunk_shape=())
    cons = HDF5Consolidator(stream_resource, descriptor)
    cons.join_method = join_method
    assert cons.shape == (0, *expected[1:])
    for i in range(5):
        doc = stream_datum_factory(data_key, i, i, i + 1)
        cons.consume_stream_datum(doc)
    assert cons.shape == expected


supported_image_seq_formats = ["jpeg", "tiff", "jpg", "tif"]


@pytest.mark.parametrize("data_key, frames_per_datum, join_method, expected", shape_testdata)
@pytest.mark.parametrize("image_format", supported_image_seq_formats)
@pytest.mark.parametrize("indx_per_stream_datum_doc", [1, 2, 3, 5])
def test_tiff_and_jpeg_shape(
    descriptor,
    image_seq_stream_resource_factory,
    stream_datum_factory,
    image_format,
    data_key,
    frames_per_datum,
    join_method,
    expected,
    indx_per_stream_datum_doc,
):
    stream_resource = image_seq_stream_resource_factory(
        image_format=image_format, data_key=data_key, chunk_shape=(1,)
    )
    cons = consolidator_factory(stream_resource, descriptor)
    cons.join_method = join_method
    assert cons.shape == (0, *expected[1:])
    for i in range(ceil(5 / indx_per_stream_datum_doc)):
        doc = stream_datum_factory(
            data_key, i, i * indx_per_stream_datum_doc, min((i + 1) * indx_per_stream_datum_doc, 5)
        )
        cons.consume_stream_datum(doc)
    assert cons.shape == expected

    # Stackable case here corresponds to multipage tiffs (AD does not support them though)
    assert len(cons.assets) == 5 * frames_per_datum if join_method == "concat" else 5


# Tuples of (data_key, chunk_shape, expected_shape, expected_chunks)
csv_testdata = [
    # 5 events, 1 or 7 array per event, 1 element in array
    ("test_arr", (1,), (5, 3), ((1,) * 5, (3,))),
    ("test_7_arrs", (1,), (35, 3), ((1,) * 35, (3,))),
    ("test_arr", (), (5, 3), ((5,), (3,))),
    ("test_7_arrs", (), (35, 3), ((35,), (3,))),
    ("test_arr", (10,), (5, 3), ((1,) * 5, (3,))),
    ("test_7_arrs", (10,), (35, 3), ((7, 7, 7, 7, 7), (3,))),
    ("test_7_arrs", (3,), (35, 3), ((3, 3, 1) * 5, (3,))),
]


@pytest.mark.parametrize("data_key, chunk_shape, expected_shape, expected_chunks", csv_testdata)
def test_csv_shape_and_chunks(
    descriptor,
    csv_stream_resource_factory,
    stream_datum_factory,
    data_key,
    chunk_shape,
    expected_shape,
    expected_chunks,
):
    stream_resource = csv_stream_resource_factory(data_key=data_key, chunk_shape=chunk_shape)
    cons = consolidator_factory(stream_resource, descriptor)
    assert cons.join_method == "concat"
    assert not cons.join_chunks
    assert cons.shape == (0, *expected_shape[1:])
    for i in range(5):
        doc = stream_datum_factory(data_key, i, i, i + 1)
        cons.consume_stream_datum(doc)
    assert cons.shape == expected_shape
    assert cons.chunks == expected_chunks


# Tuples of (data_key, join_method, join_chunks, chunk_shape, expected_chunks)
chunk_hdf5_testdata = [
    ("test_img", "stack", True, (), ((5,), (1,), (10,), (15,))),
    ("test_img", "stack", True, (1, 1, 10, 15), ((1, 1, 1, 1, 1), (1,), (10,), (15,))),
    ("test_img", "stack", True, (2,), ((2, 2, 1), (1,), (10,), (15,))),
    ("test_img", "stack", True, (5, 1, 10, 15), ((5,), (1,), (10,), (15,))),
    ("test_img", "stack", True, (10, 1), ((5,), (1,), (10,), (15,))),
    ("test_img", "stack", True, (3, 1, 4, 5), ((3, 2), (1,), (4, 4, 2), (5, 5, 5))),
    ("test_7_imgs", "stack", True, (), ((5,), (7,), (10,), (15,))),
    ("test_7_imgs", "stack", True, (1, 1), ((1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1, 1), (10,), (15,))),
    ("test_7_imgs", "stack", True, (2,), ((2, 2, 1), (7,), (10,), (15,))),
    ("test_7_imgs", "stack", True, (5, 1, 10, 15), ((5,), (1, 1, 1, 1, 1, 1, 1), (10,), (15,))),
    ("test_7_imgs", "stack", True, (10, 5), ((5,), (5, 2), (10,), (15,))),
    (
        "test_7_imgs",
        "stack",
        True,
        (3, 4, 5, 6),
        (
            (3, 2),
            (4, 3),
            (5, 5),
            (6, 6, 3),
        ),
    ),
    ("test_cube", "stack", True, (3, 1, 4, 5, 3), ((3, 2), (1,), (4, 4, 2), (5, 5, 5), (3,))),
    ("test_7_cubes", "stack", True, (3, 4, 5, 6, 7), ((3, 2), (4, 3), (5, 5), (6, 6, 3), (3,))),
    ("test_arr", "stack", True, (5, 1, 1), ((5,), (1,), (1, 1, 1))),
    ("test_arr", "stack", True, (2,), ((2, 2, 1), (1,), (3,))),
    ("test_7_arrs", "stack", True, (5, 1, 1), ((5,), (1, 1, 1, 1, 1, 1, 1), (1, 1, 1))),
    ("test_7_arrs", "stack", True, (2,), ((2, 2, 1), (7,), (3,))),
    ("test_num", "stack", True, (), ((5,), (1,))),
    ("test_num", "stack", True, (2,), ((2, 2, 1), (1,))),
    ("test_7_nums", "stack", True, (), ((5,), (7,))),
    ("test_7_nums", "stack", True, (2,), ((2, 2, 1), (7,))),
    ("test_7_nums", "stack", True, (2, 3), ((2, 2, 1), (3, 3, 1))),
    ("test_img", "concat", True, (), ((5,), (10,), (15,))),
    ("test_img", "concat", True, (1, 10, 15), ((1, 1, 1, 1, 1), (10,), (15,))),
    ("test_img", "concat", True, (2,), ((2, 2, 1), (10,), (15,))),
    ("test_img", "concat", True, (5, 10, 15), ((5,), (10,), (15,))),
    ("test_img", "concat", True, (10, 1), ((5,), (1,) * 10, (15,))),
    ("test_img", "concat", True, (3, 4, 5), ((3, 2), (4, 4, 2), (5, 5, 5))),
    ("test_7_imgs", "concat", True, (), ((35,), (10,), (15,))),
    ("test_7_imgs", "concat", False, (), ((35,), (10,), (15,))),
    ("test_7_imgs", "concat", True, (1, 1), ((1,) * 35, (1,) * 10, (15,))),
    ("test_7_imgs", "concat", False, (2,), ((2, 2, 2, 1) * 5, (10,), (15,))),
    ("test_7_imgs", "concat", True, (2,), ((2,) * 17 + (1,), (10,), (15,))),
    ("test_7_imgs", "concat", True, (5, 10, 15), ((5,) * 7, (10,), (15,))),
    ("test_7_imgs", "concat", False, (5, 10, 15), ((5, 2) * 5, (10,), (15,))),
    ("test_7_imgs", "concat", True, (10, 5), ((10, 10, 10, 5), (5, 5), (15,))),
    ("test_7_imgs", "concat", False, (10, 5), ((7,) * 5, (5, 5), (15,))),
    (
        "test_7_imgs",
        "concat",
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
        "concat",
        False,
        (3, 5, 6),
        (
            (3, 3, 1) * 5,
            (5, 5),
            (6, 6, 3),
        ),
    ),
    ("test_cube", "concat", True, (3, 4, 5, 3), ((3, 2), (4, 4, 2), (5, 5, 5), (3,))),
    ("test_7_cubes", "concat", True, (3, 5, 6, 7), ((3,) * 11 + (2,), (5, 5), (6, 6, 3), (3,))),
    ("test_7_cubes", "concat", False, (3, 5, 6, 7), ((3, 3, 1) * 5, (5, 5), (6, 6, 3), (3,))),
    ("test_arr", "concat", True, (5, 1), ((5,), (1, 1, 1))),
    ("test_arr", "concat", True, (2,), ((2, 2, 1), (3,))),
    ("test_7_arrs", "concat", True, (5, 1), ((5,) * 7, (1, 1, 1))),
    ("test_7_arrs", "concat", False, (5, 1), ((5, 2) * 5, (1, 1, 1))),
    ("test_7_arrs", "concat", True, (2,), ((2,) * 17 + (1,), (3,))),
    ("test_7_arrs", "concat", False, (2,), ((2, 2, 2, 1) * 5, (3,))),
    ("test_num", "concat", True, (), ((5,),)),
    ("test_num", "concat", True, (2,), ((2, 2, 1),)),
    ("test_7_nums", "concat", True, (), ((35,),)),
    ("test_7_nums", "concat", False, (), ((35,),)),
    ("test_7_nums", "concat", True, (3,), ((3,) * 11 + (2,),)),
    ("test_7_nums", "concat", False, (3,), ((3, 3, 1) * 5,)),
]


@pytest.mark.parametrize("data_key, join_method, join_chunks, chunk_shape, expected", chunk_hdf5_testdata)
def test_hdf5_chunks(
    descriptor,
    hdf5_stream_resource_factory,
    stream_datum_factory,
    data_key,
    join_method,
    join_chunks,
    chunk_shape,
    expected,
):
    stream_resource = hdf5_stream_resource_factory(data_key=data_key, chunk_shape=chunk_shape)
    cons = HDF5Consolidator(stream_resource, descriptor)
    cons.join_method = join_method
    cons.join_chunks = join_chunks
    assert cons.chunks == ((0,), *expected[1:])
    for i in range(5):
        doc = stream_datum_factory(data_key, i, i, i + 1)
        cons.consume_stream_datum(doc)
    assert cons.chunks == expected


# Tuples of (data_key, join_method, join_chunks, frames_per_datum, indx_per_stream_datum_doc, chunk_shape, expected_chunks) # noqa
chunk_tiff_testdata = [
    ("test_img", "stack", True, 1, 1, (1,), ((1, 1, 1, 1, 1), (1,), (10,), (15,))),
    ("test_img", "stack", True, 1, 2, (1,), ((1, 1, 1, 1, 1), (1,), (10,), (15,))),
    ("test_img", "stack", True, 1, 5, (1,), ((1, 1, 1, 1, 1), (1,), (10,), (15,))),
    ("test_6_imgs", "stack", True, 6, 1, (1,), ((1,) * 5, (6,), (10,), (15,))),
    ("test_6_imgs", "concat", True, 6, 1, (1,), ((1,) * 30, (10,), (15,))),
    ("test_6_imgs", "stack", True, 6, 2, (2,), ((2, 2, 1), (6,), (10,), (15,))),
    ("test_6_imgs", "concat", True, 6, 2, (2,), ((2,) * 15, (10,), (15,))),
    ("test_6_imgs", "stack", True, 6, 4, (3,), ((3, 2), (6,), (10,), (15,))),
    ("test_6_imgs", "concat", True, 6, 4, (3,), ((3,) * 10, (10,), (15,))),
    ("test_6_imgs", "stack", True, 6, 1, (5,), None),  # chunk_shape[0] must devide the number of frames
    ("test_6_imgs", "concat", True, 6, 1, (5,), None),
    ("test_6_imgs", "concat", True, 6, 10, (10,), None),
]


@pytest.mark.parametrize(
    "data_key, join_method, join_chunks, frames_per_datum, indx_per_stream_datum_doc, chunk_shape, expected_chunks",  # noqa
    chunk_tiff_testdata,
)
@pytest.mark.parametrize("image_format", supported_image_seq_formats)
def test_tiff_and_jpeg_chunks(
    descriptor,
    image_seq_stream_resource_factory,
    stream_datum_factory,
    image_format,
    data_key,
    join_method,
    join_chunks,
    frames_per_datum,
    indx_per_stream_datum_doc,
    chunk_shape,
    expected_chunks,
):
    """Test the chunking of (possibly multipage) tiff and jpeg datasets and the number of registered files."""

    stream_resource = image_seq_stream_resource_factory(
        image_format=image_format, data_key=data_key, chunk_shape=chunk_shape
    )
    if expected_chunks is None:
        with pytest.raises(AssertionError):
            cons = consolidator_factory(stream_resource, descriptor)
        return

    cons = consolidator_factory(stream_resource, descriptor)
    cons.join_method = join_method
    cons.join_chunks = join_chunks
    assert cons.chunks == ((0,), *expected_chunks[1:])
    for i in range(ceil(5 / indx_per_stream_datum_doc)):
        doc = stream_datum_factory(
            data_key, i, i * indx_per_stream_datum_doc, min((i + 1) * indx_per_stream_datum_doc, 5)
        )
        cons.consume_stream_datum(doc)
    assert cons.chunks == expected_chunks

    # Check the number of registered files
    assert len(cons.assets) == 5 * frames_per_datum / expected_chunks[0][0] if join_method == "concat" else 5


template_testdata = [
    ("", "img_{:06d}", "img_{:06d}", "img_000042"),
    ("img", "{:s}_{:06d}", "img_{:06d}", "img_000042"),
    ("img", "%s_%06d", "img_{:06d}", "img_000042"),
    ("", "img%s_%06d", "img_{:06d}", "img_000042"),
    ("img", "%s_%1d", "img_{:1d}", "img_42"),
    ("img", "%s_%-6d", "img_{:<6d}", "img_42    "),
    ("img", "%s_%+06d", "img_{:+06d}", "img_+00042"),
    ("img", "%s_% 06d", "img_{: 06d}", "img_ 00042"),
    ("img", "%s_%-+6d", "img_{:<+6d}", "img_+42   "),
    ("img", "%s_%- 6d", "img_{:< 6d}", "img_ 42   "),
]


@pytest.mark.parametrize("image_format", supported_image_seq_formats)
@pytest.mark.parametrize("filename, original_template, expected_template, formatted", template_testdata)
def test_name_templating(
    descriptor,
    image_seq_stream_resource_factory,
    image_format,
    filename,
    original_template,
    expected_template,
    formatted,
):
    stream_resource = image_seq_stream_resource_factory(
        image_format=image_format, data_key="test_img", chunk_shape=((1),)
    )
    stream_resource["parameters"]["template"] = f"{original_template}.{image_format}"
    if filename:
        stream_resource["parameters"]["filename"] = filename
    cons = consolidator_factory(stream_resource, descriptor)
    assert cons.template == f"{expected_template}.{image_format}"
    assert cons.template.format(42) == f"{formatted}.{image_format}"
