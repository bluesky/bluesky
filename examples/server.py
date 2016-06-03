from bluesky.run_engine import RunEngineNamedLookup
from bluesky.examples import motor
from bluesky.service_server import start_run_engine


def main():
    RE = RunEngineNamedLookup({'motor': motor})
    RE.msg_hook = print
    start_run_engine(RE)


if __name__ == "__main__":
    main()
