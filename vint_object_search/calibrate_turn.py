"""Drive the rover at a fixed command to calibrate config.yaml.

Turning: mark the heading, run `python calibrate_turn.py --angular 0.5 --seconds 3`, and
enter the degrees it turned. Driving: run `python calibrate_turn.py --linear 0.3 --seconds 3`
and enter the inches it traveled. Pass both to measure turning while driving, e.g.
`--linear 0.3 --angular 0.3`. Copy the printed values into the sdk section.
"""
import argparse
import math
import time
from pathlib import Path

import yaml

from object_search_vint import send_control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--angular", type=float, help="control units; positive is left")
    parser.add_argument("--linear", type=float, help="control units; positive is forward")
    parser.add_argument("--seconds", type=float, default=3.0)
    arguments = parser.parse_args()

    with arguments.config.open() as file:
        sdk = yaml.safe_load(file)["sdk"]
    base_url = sdk["base_url"]
    interval = 1.0 / sdk["control_hz"]

    linear = arguments.linear or 0.0
    angular = arguments.angular if arguments.angular is not None else (
        0.0 if arguments.linear is not None else 0.5)
    print(f"Driving at linear={linear:+.2f} angular={angular:+.2f} for {arguments.seconds:.1f}s")
    deadline = time.monotonic() + arguments.seconds
    try:
        while time.monotonic() < deadline:
            send_control(base_url, linear, angular)
            time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
    finally:
        send_control(base_url, 0.0, 0.0)

    if linear and angular:
        degrees = float(input("Degrees turned (measured): "))
        print(f"\nTurn rate while driving at angular={angular}: "
              f"{math.radians(degrees) / arguments.seconds:.3f} rad/s")
    elif linear:
        inches = float(input("Inches traveled (measured): "))
        speed = inches * 0.0254 / arguments.seconds
        print(f"\nsdk:\n  max_linear: {abs(linear)}\n  max_v_mps: {speed:.3f}")
    else:
        degrees = float(input("Degrees turned (measured): "))
        rate = math.radians(degrees) / arguments.seconds
        print(f"\nsdk:\n  max_angular: {abs(angular)}\n  max_w_radps: {rate:.3f}")


if __name__ == "__main__":
    main()
