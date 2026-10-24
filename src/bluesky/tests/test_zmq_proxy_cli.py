from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture

from bluesky.callbacks.zmq import ClientCurve, ServerCurve
from bluesky.commandline.zmq_proxy import build_client_curve, build_server_curve, start_dispatcher
from bluesky.commandline.zmq_proxy import main as zmq_proxy_main


@pytest.fixture
def sample_args():
    return SimpleNamespace(
        **{
            "in_port": None,
            "out_port": None,
            "out_address": "tcp://localhost:5556",
            "in_address": "tcp://localhost:5555",
            "in_mode": "bind",
            "out_mode": "connect",
            "in_client_secret": None,
            "in_server_public": None,
            "in_curve_secret": None,
            "in_curve_client_keys": None,
            "in_curve_allow": None,
            "out_client_secret": None,
            "out_server_public": None,
            "out_curve_secret": None,
            "out_curve_client_keys": None,
            "out_curve_allow": None,
            "verbose": False,
            "logfile": None,
        }
    )


@pytest.mark.parametrize("with_curve", [True, False])
def test_zmq_proxy_start_dispatcher(mocker: MockerFixture, with_curve: bool, tmp_path: Path):

    curve = ClientCurve(tmp_path, tmp_path) if with_curve else None
    mock_dispatcher_cls = mocker.patch("bluesky.commandline.zmq_proxy.RemoteDispatcher")
    mock_dispatcher_instance = mock_dispatcher_cls.return_value
    start_dispatcher("tcp://localhost:5555", curve=curve)
    mock_dispatcher_cls.assert_called_once_with("tcp://localhost:5555", curve_config=curve)
    mock_dispatcher_instance.subscribe.assert_called_once()
    mock_dispatcher_instance.start.assert_called_once()


@pytest.mark.parametrize(
    ("client_keys", "allow"), [("client_keys", None), (None, ["allow"]), ("client_keys", ["allow"])]
)
def test_zmq_proxy_build_server_curve_invalid(client_keys, allow):
    with pytest.raises(ValueError, match="Cannot specify client_keys or allow without providing a secret key"):
        build_server_curve(secret=None, client_keys=client_keys, allow=allow)


@pytest.mark.parametrize(
    ("secret_path", "client_keys", "allow", "expected_secret_path", "expected_client_keys", "expected_allow"),
    [
        (None, None, None, None, None, None),
        ("secret.key", None, None, Path("secret.key"), None, None),
        ("secret.key", "client_keys", None, Path("secret.key"), Path("client_keys"), None),
        ("secret.key", None, ["allow", "allow"], Path("secret.key"), None, {"allow"}),
    ],
)
def test_zmq_proxy_build_server_curve(
    secret_path: str, client_keys, allow, expected_secret_path, expected_client_keys, expected_allow
):
    curve = build_server_curve(secret=secret_path, client_keys=client_keys, allow=allow)
    if secret_path is None:
        assert curve is None
        return
    assert isinstance(curve, ServerCurve)
    assert curve.secret_path == expected_secret_path
    assert curve.client_public_keys == expected_client_keys
    assert curve.allow == expected_allow


@pytest.mark.parametrize(
    ("secret", "server_public"),
    [
        ("secret.key", None),
        (None, "server_public.key"),
    ],
)
def test_build_client_curve_invalid(secret, server_public):
    with pytest.raises(
        ValueError, match="Both client secret and server public key must be provided for CURVE client mode"
    ):
        build_client_curve(secret=secret, server_public=server_public)


@pytest.mark.parametrize(
    ("secret", "server_public"),
    [
        ("secret.key", "server_public.key"),
        (None, None),
    ],
)
def test_build_client_curve_valid(secret, server_public):
    curve = build_client_curve(secret=secret, server_public=server_public)
    if secret is None and server_public is None:
        assert curve is None
        return
    assert isinstance(curve, ClientCurve)
    assert curve.secret_path == Path(secret)
    assert curve.server_public_key == Path(server_public)


@pytest.mark.parametrize("in_or_out", ["in", "out"])
def test_zmq_proxy_main_both_inport_outport(mocker: MockerFixture, in_or_out: str):
    mocker.patch(
        "argparse.ArgumentParser.parse_args",
        return_value=SimpleNamespace(
            **{f"{in_or_out}_port": "tcp://localhost:5555", f"{'out' if in_or_out == 'in' else 'in'}_port": None}
        ),
    )
    with pytest.raises(ValueError, match="Both in_port and out_port positional arguments must be provided."):
        zmq_proxy_main()


@pytest.mark.parametrize(
    "out_addr,in_addr",
    [
        ("tcp://localhost:5555", "tcp://localhost:5556"),
        ("tcp://localhost:5555", None),
        (None, "tcp://localhost:5556"),
    ],
)
def test_zmq_proxy_main_only_inport_or_outport(mocker: MockerFixture, out_addr: str, in_addr: str):
    args = {
        "in_port": "tcp://localhost:5555",
        "out_port": "tcp://localhost:5556",
        "out_address": out_addr,
        "in_address": in_addr,
    }
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=SimpleNamespace(**args))
    with pytest.raises(
        ValueError, match=r"Cannot mix positional arguments \(in_port, out_port\) with optional arguments"
    ):
        zmq_proxy_main()


@pytest.mark.parametrize(
    ("in_or_out", "client_secret", "server_public"),
    [
        (in_or_out, in_client_secret, in_server_public)
        for in_or_out in ["in", "out"]
        for in_client_secret, in_server_public in [
            ("client_secret.key", "server_public.key"),
            ("client_secret.key", None),
            (None, "server_public.key"),
        ]
    ],
)
def test_zmq_proxy_main_cannot_bind_client_mode(
    mocker: MockerFixture, in_or_out: str, client_secret: str, server_public: str
):
    args = {
        "in_port": None,
        "out_port": None,
        f"{in_or_out}_mode": "bind",
        f"{'out' if in_or_out == 'in' else 'in'}_mode": "connect",
        "in_address": "tcp://localhost:5555",
        "out_address": "tcp://localhost:5556",
        f"{in_or_out}_client_secret": client_secret,
        f"{in_or_out}_server_public": server_public,
        "in_curve_secret": None,
        "in_curve_client_keys": None,
        "in_curve_allow": None,
    }
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=SimpleNamespace(**args))
    with pytest.raises(
        ValueError,
        match="Cannot use client CURVE options",
    ):
        zmq_proxy_main()


@pytest.mark.parametrize(
    ("in_or_out", "curve_secret", "curve_client_keys", "curve_allow"),
    [
        (in_or_out, curve_secret, curve_client_keys, curve_allow)
        for in_or_out in ["in", "out"]
        for curve_secret, curve_client_keys, curve_allow in [
            ("curve_secret.key", ["client1.key"], ["192.168.1.1"]),
            ("curve_secret.key", ["client1.key"], None),
            ("curve_secret.key", None, ["192.168.1.1"]),
            ("curve_secret.key", None, None),
        ]
    ],
)
def test_zmq_proxy_main_cannot_connect_server_mode(
    mocker: MockerFixture,
    in_or_out: str,
    curve_secret: str,
    curve_client_keys: list[str] | None,
    curve_allow: list[str] | None,
):
    args = {
        "in_port": None,
        "out_port": None,
        f"{in_or_out}_mode": "connect",
        f"{'out' if in_or_out == 'in' else 'in'}_mode": "bind",
        "in_address": "tcp://localhost:5555",
        "out_address": "tcp://localhost:5556",
        "in_client_secret": None,
        "in_server_public": None,
        f"{in_or_out}_curve_secret": curve_secret,
        f"{in_or_out}_curve_client_keys": curve_client_keys,
        f"{in_or_out}_curve_allow": curve_allow,
    }
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=SimpleNamespace(**args))
    with pytest.raises(
        ValueError,
        match="Cannot use server CURVE options",
    ):
        zmq_proxy_main()


def test_zmq_proxy_main_successful_startup(mocker: MockerFixture, sample_args: SimpleNamespace):

    mocker.patch("argparse.ArgumentParser.parse_args", return_value=sample_args)
    mock_proxy_cls = mocker.patch("bluesky.commandline.zmq_proxy.Proxy")
    # mocker.patch("bluesky.commandline.zmq.Proxy.start", return_value=None)
    mock_proxy_instance = mock_proxy_cls.return_value
    mock_proxy_instance.start.return_value = None
    zmq_proxy_main()
    mock_proxy_instance.start.assert_called_once()


def test_zmq_proxy_main_successful_startup_with_curve(
    mocker: MockerFixture, sample_args: SimpleNamespace, tmp_path: Path
):

    sample_args.in_curve_secret = str(tmp_path / "in_secret.key")
    sample_args.in_curve_client_keys = str(tmp_path / "client_keys")
    sample_args.in_curve_allow = ["192.168.1.1"]
    sample_args.in_mode = "bind"
    sample_args.out_mode = "connect"
    sample_args.out_server_public = str(tmp_path / "out_server_public.key")
    sample_args.out_client_secret = str(tmp_path / "out_client_secret.key")
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=sample_args)
    mock_proxy_cls = mocker.patch("bluesky.commandline.zmq_proxy.Proxy")
    mock_proxy_instance = mock_proxy_cls.return_value
    mock_proxy_instance.start.return_value = None
    zmq_proxy_main()
    mock_proxy_cls.assert_called_once_with(
        sample_args.in_address,
        sample_args.out_address,
        in_curve=ServerCurve(
            secret_path=Path(sample_args.in_curve_secret),
            client_public_keys=Path(sample_args.in_curve_client_keys),
            allow=set(sample_args.in_curve_allow),
        ),
        out_curve=ClientCurve(
            secret_path=Path(sample_args.out_client_secret), server_public_key=Path(sample_args.out_server_public)
        ),
        in_bind=sample_args.in_mode == "bind",
        out_bind=sample_args.out_mode == "bind",
    )
    mock_proxy_instance.start.assert_called_once()


def test_zmq_proxy_main_valid_startup_with_verbose(mocker: MockerFixture, sample_args: SimpleNamespace):

    sample_args.verbose = True
    sample_args.out_mode = "bind"
    mocker.patch("argparse.ArgumentParser.parse_args", return_value=sample_args)
    mock_thread = mocker.patch("threading.Thread")
    mock_proxy_cls = mocker.patch("bluesky.commandline.zmq_proxy.Proxy")
    mock_proxy_instance = mock_proxy_cls.return_value
    mock_proxy_instance.start.return_value = None
    zmq_proxy_main()
    mock_proxy_instance.start.assert_called_once()
    mock_thread.assert_called_once_with(
        target=start_dispatcher, args=(mock_proxy_instance.out_port, None), daemon=True
    )
