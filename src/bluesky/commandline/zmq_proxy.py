import argparse
import logging
import threading
from pathlib import Path

from bluesky.callbacks.zmq import Proxy, RemoteDispatcher, ServerCurve, ClientCurve

logger = logging.getLogger("bluesky")


def start_dispatcher(out_address, curve, logfile=None):
    """The dispatcher function
    Parameters
    ----------
    logfile : string
        string come from user command. ex --logfile=temp.log
        logfile will be "temp.log". logfile could be empty.
    """
    dispatcher = RemoteDispatcher(out_address, curve_config=curve)
    if logfile is not None:
        raise ValueError(
            "Parameter 'logfile' is deprecated and will be removed in future releases. "
            "Currently it does not have effect. Call the function with 'logfile=None' "
        )

    def log_writer(name, doc):
        """logger's wrapper function
        This function will be used to fit .subscribe() method.
        It has two arguments as .subscribe expects. Inside, it
        calls logger.* to write doc which is a dict as a str
        into logfile
        """
        if name in ("start", "stop"):
            logger.info("%s: %r", name, doc)
        else:
            logger.debug("%s: %r", name, doc)

    dispatcher.subscribe(log_writer)  # Subscribe log writer
    dispatcher.start()


def main():
    DESC = "Start a 0MQ proxy for publishing bluesky documents over a network."
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument("--in-address", help="port that RunEngines should broadcast to")
    parser.add_argument("--out-address", help="port that subscribers should subscribe to")
    # CURVE security options for input socket
    parser.add_argument("--in-curve-secret", type=str, help="Path to CURVE server secret key for input socket")
    parser.add_argument(
        "--in-curve-client-keys", type=str, help="Path to folder of client public keys for input socket"
    )
    parser.add_argument(
        "--in-curve-allow", type=str, nargs="*", help="Set of IP addresses to allow for input socket"
    )
    # CURVE security options for output socket
    parser.add_argument("--out-curve-secret", type=str, help="Path to CURVE server secret key for output socket")
    parser.add_argument(
        "--out-curve-client-keys", type=str, help="Path to folder of client public keys for output socket"
    )
    parser.add_argument(
        "--out-curve-allow", type=str, nargs="*", help="Set of IP addresses to allow for output socket"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        help=("Show 'start' and 'stop' documents. (Use -vvv to show all documents.)"),
    )
    parser.add_argument("--logfile", type=str, help="Redirect logging output to a file on disk.")
    args = parser.parse_args()

    # Helper to build ServerCurve or None
    def build_server_curve(
        secret: str | None, client_keys: str | None, allow: list[str] | None
    ) -> ServerCurve | None:
        if secret is None:
            if client_keys is not None or allow is not None:
                raise ValueError("Cannot specify client_keys or allow without providing a secret key")
            return None
        secret_path = Path(secret)
        client_public_keys = Path(client_keys) if client_keys else None
        allow_set = set(allow) if allow else None
        return ServerCurve(secret_path=secret_path, client_public_keys=client_public_keys, allow=allow_set)

    in_curve = build_server_curve(args.in_curve_secret, args.in_curve_client_keys, args.in_curve_allow)
    out_curve = build_server_curve(args.out_curve_secret, args.out_curve_client_keys, args.out_curve_allow)

    # Configure logging BEFORE creating the proxy so we capture socket configuration debug messages
    if args.verbose:
        from bluesky.log import config_bluesky_logging
        import bluesky.log

        # "INFO" if called with '-v' or '-vv', "DEBUG" if called with '-vvv'
        level = "INFO" if args.verbose <= 2 else "DEBUG"
        if args.logfile:
            config_bluesky_logging(level=level, file=args.logfile)
        else:
            print(f"configuring blueskylogging to {level}")
            config_bluesky_logging(level=level)
        logging.getLogger("zmq").setLevel(level)
        logging.getLogger("zmq").addHandler(bluesky.log.current_handler)

    print("Connecting...")
    try:
        in_address = int(args.in_address)
    except (ValueError, TypeError):
        in_address = args.in_address
    try:
        out_address = int(args.out_address)
    except (ValueError, TypeError):
        out_address = args.out_address
    proxy = Proxy(in_address, out_address, in_curve=in_curve, out_curve=out_curve)
    print("Receiving on address %s; publishing to address %s." % (proxy.in_port, proxy.out_port))
    if args.verbose:
        # Set daemon to kill all threads upon IPython exit
        if out_curve is None:
            # We would need client certificates setup to connect to the output port
            client_curve = None
        else:
            # this looks funny, but the secret file also contains the public key
            # this bets that the public key for the server is in the folder of public keys
            # it will accept and that we can route to the output port on an allowed ip
            client_curve = ClientCurve(out_curve.secret_path, out_curve.secret_path)
        threading.Thread(target=start_dispatcher, args=(proxy.out_port, client_curve), daemon=True).start()


    print("Use Ctrl+C to exit.")
    try:
        proxy.start()
    except KeyboardInterrupt:
        print("Interrupted. Exiting...")


if __name__ == "__main__":
    main()
