from bluesky.service_client import execute_on_remote
from bluesky.plans import jsonify
from bluesky.plans import scan
from bluesky.examples import motor

def main():
    plan = scan([], motor, 1, 3, 3)
    execute_on_remote(jsonify(plan))


if __name__ == "__main__":
    main()
