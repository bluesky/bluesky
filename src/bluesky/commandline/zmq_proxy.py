import argparse
import logging
import threading

from bluesky.callbacks.zmq import Proxy, RemoteDispatcher

logger = logging.getLogger("bluesky")


def start_dispatcher(out_address, logfile=None):
    """The dispatcher function
    Parameters
    ----------
    logfile : string
        string come from user command. ex --logfile=temp.log
        logfile will be "temp.log". logfile could be empty.
    """
    dispatcher = RemoteDispatcher(out_address)
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
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        help=("Show 'start' and 'stop' documents. (Use -vvv to show all documents.)"),
    )
    parser.add_argument("--logfile", type=str, help="Redirect logging output to a file on disk.")
    args = parser.parse_args()

    print("Connecting...")
    proxy = Proxy(args.in_address, args.out_address)
    print("Receiving on address %s; publishing to address %s." % (proxy.in_port, proxy.out_port))
    if args.verbose:
        from bluesky.log import config_bluesky_logging

        # "INFO" if called with '-v' or '-vv', "DEBUG" if called with '-vvv'
        level = "INFO" if args.verbose <= 2 else "DEBUG"
        if args.logfile:
            config_bluesky_logging(level=level, file=args.logfile)
        else:
            config_bluesky_logging(level=level)
        # Set daemon to kill all threads upon IPython exit
        threading.Thread(target=start_dispatcher, args=(proxy.out_port,), daemon=True).start()

    print("Use Ctrl+C to exit.")
    try:
        proxy.start()
    except KeyboardInterrupt:
        print("Interrupted. Exiting...")


if __name__ == "__main__":
    main()
