import os
import argparse
import operator
import shutil

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import fsolve

from helicopter import Helicopter
from mission import Mission
from utilities import (
    PI,
    RotorBlade,
    color,
    display_text,
    emperical_function,
    g,
    parse_toml_params,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Simulate the helicopter mission.""")
    parser.add_argument(
        "--hel_params",
        type=str,
        default="data/params.toml",
        help="path/to/the/helicopter/parameters/file.toml",
    )
    parser.add_argument(
        "--mission_params",
        type=str,
        default="data/individual_mission_A.toml",
        help="path/to/the/mission/parameters/file.toml",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./images/individual_mission_A/",
        help="path/to/the/output/directory",
    )

    args = parser.parse_args()

    design_params = parse_toml_params(args.hel_params)
    mission_params = parse_toml_params(args.mission_params)
    mission_params["plot"] = False

    os.makedirs(args.output_dir, exist_ok=True)

    for altitude in range(9000, 9200, 10):
        mission_params["mission"]["2"]["altitude"] = altitude
        mission = Mission(design_params, mission_params, args.output_dir)
        try:
            mission.simulate()
            print(mission.fuel)
        except ValueError as e:
            print(f"Mission failed at altitude: {altitude} due to {e}")
            break

    mission_params["plot"] = True
    mission_params["mission"]["2"]["altitude"] = altitude - 10
    mission = Mission(design_params, mission_params, args.output_dir)
    mission.simulate()
