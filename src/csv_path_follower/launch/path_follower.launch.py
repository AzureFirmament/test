#! /usr/bin/env python3
"""
Place in: csv_path_follower/launch/path_follower.launch.py

    bl csv_path_follower path_follower.launch.py is_sim:=True
    bl csv_path_follower path_follower.launch.py is_sim:=False map_name:=tum_d_floor3
"""

import os
from ament_index_python.packages import get_package_share_directory
from better_launch import BetterLaunch, launch_this

REAL_NAME = "self"
SIM_NAME = "svea_a"


@launch_this
def main(
    is_sim: bool = True,
    map_name: str = "sml",
    use_foxglove: bool = True,
    path_csv: str = "",
    target_speed: float = 0.35,
    # Start the car on the first point of the path, heading along it.
    initial_pose_x: float = 1.0,
    initial_pose_y: float = -2.5,
    initial_pose_a: float = 1.57,
):
    bl = BetterLaunch()

    if not path_csv:
        path_csv = os.path.join(
            get_package_share_directory("csv_path_follower"),
            "paths", "tiha_path_data.csv")

    name = SIM_NAME if is_sim else REAL_NAME

    bl.include("svea_core", "svea.launch.py",
               name=name,
               is_sim=is_sim,
               map_name=map_name,
               initial_pose_x=initial_pose_x,
               initial_pose_y=initial_pose_y,
               initial_pose_a=initial_pose_a)

    with bl.group(name):
        bl.node("csv_path_follower", "pure_pursuit_tracking.py",
                name="path_follower",
                params={
                    "path_csv": path_csv,
                    "target_speed": target_speed,
                    "localization/base_frame": f"{name}/base_link",
                })

    if use_foxglove:
        bl.include("foxglove_bridge", "foxglove_bridge_launch.xml")