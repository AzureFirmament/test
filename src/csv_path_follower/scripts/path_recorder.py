#! /usr/bin/env python3
"""
Record the actually-driven path (from mocap) while the CSV follower runs.

Place in: csv_path_follower/scripts/path_recorder.py

Hardware only: this node is launched by path_follower.launch.py exclusively
when is_sim:=False. In simulation there is no mocap feed and no recorder.

Output: one CSV per run in `output_dir`, columns

    timestamp, x, y, yaw, velocity, steering

All rows are RAW mocap coordinates -- the same frame as the input path CSV
(path_frame, e.g. 'mocap'), so a recording overlays directly on the
reference path with no conversion.

Lifecycle
---------

    start   Bool True on start_topic ('/csv_follower' -- the same signal that
            releases the follower's throttle)         --> open a new file
    stop    Bool True on done_topic ('path_done', published latched by the
            follower when the last segment completes OR on abort),
            or Bool False on start_topic,
            or node shutdown                          --> close the file

A new True on start_topic after a stop opens a fresh, timestamped file, so
nothing is ever overwritten.

Signals
-------

    pose        raw mocap PoseStamped on mocap_pose_topic; one row per
                message (thin out with min_record_dt if needed).
    velocity    longitudinal component of the finite-differenced mocap
                position, projected onto the heading, so it is SIGNED
                (negative while reversing). EMA-smoothed with vel_smoothing.
    steering    the follower's commanded pure pursuit angle, mirrored on
                ctrl_topic (TwistStamped: angular.z = steering [rad],
                linear.x = signed commanded speed [m/s]). Swap in the
                commanded speed instead of the measured one in _write() if
                you prefer.

Topic name conventions (matters!): ctrl_topic and done_topic are RELATIVE so
that bl.group(name) namespaces them to /<name>/..., matching the follower in
the same group. start_topic is absolute, matching the follower's own
'/csv_follower' subscription.
"""

import csv
import math
import os
import time

from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)

from svea_core import rosonic as rx

qos_subber = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=10,
)

# Matches the follower's latched 'path_done' publisher, so a done flag
# published a moment before this node's subscription resolved is still seen.
qos_latched = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    depth=1,
)


def quaternion_to_yaw(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class path_recorder(rx.Node):
    """Log timestamp, x, y, yaw, velocity, steering while following a path."""

    # ---------------------------------------------------------------- params

    mocap_pose_topic = rx.Parameter("/svea7/pose")
    ctrl_topic = rx.Parameter("ctrl_info")      # relative -> /<name>/ctrl_info
    start_topic = rx.Parameter("/csv_follower")
    done_topic = rx.Parameter("path_done")      # relative -> /<name>/path_done

    output_dir = rx.Parameter("/tmp/path_logs")
    file_prefix = rx.Parameter("mocap_path")

    # Thin out the rows: minimum time between writes [s]. 0 = one row per
    # mocap message (~100 Hz on Qualisys).
    min_record_dt = rx.Parameter(0.0)

    # EMA factor for the mocap-differenced velocity, in (0, 1]. 1.0 = raw.
    # Mocap position noise differentiated at ~100 Hz is spiky; 0.3 keeps the
    # signal responsive at 0.35 m/s while killing most of the spikes.
    vel_smoothing = rx.Parameter(0.3)

    # ---------------------------------------------------------------- startup

    # NOTE: all subscriptions are created in on_startup, not with decorators.
    # Decorator arguments are evaluated at class-definition time, when the
    # topic parameters are still rx.Parameter descriptors -- the node would
    # come up subscribed to the wrong names, silently. (Same trap as in
    # pure_pursuit_tracking.py.)

    def on_startup(self):
        self.recording = False
        self._file = None
        self._writer = None
        self._path = ""
        self._rows = 0

        # latest commanded steering / speed from the follower
        self._steer = 0.0
        self._cmd_speed = 0.0

        # finite-difference state for the mocap velocity
        self._t_prev = None
        self._x_prev = 0.0
        self._y_prev = 0.0
        self._v_ema = 0.0

        self._last_write_t = -1.0

        os.makedirs(self.output_dir, exist_ok=True)

        self.create_subscription(
            Bool, self.start_topic, self._on_start, qos_subber)
        self.create_subscription(
            Bool, self.done_topic, self._on_done, qos_latched)
        self.create_subscription(
            TwistStamped, self.ctrl_topic, self._on_ctrl, 10)
        self.create_subscription(
            PoseStamped, self.mocap_pose_topic, self._on_mocap, qos_subber)

        self.get_logger().info(
            f"armed: raw mocap poses from '{self.mocap_pose_topic}', "
            f"waiting for True on '{self.start_topic}'; files go to "
            f"'{self.output_dir}'")

    # -------------------------------------------------------------- callbacks

    def _on_start(self, msg):
        if msg.data and not self.recording:
            self._open()
        elif not msg.data and self.recording:
            self._close(f"'{self.start_topic}' went False")

    def _on_done(self, msg):
        if msg.data and self.recording:
            self._close("path finished (done signal from follower)")

    def _on_ctrl(self, msg):
        self._steer = float(msg.twist.angular.z)
        self._cmd_speed = float(msg.twist.linear.x)

    def _on_mocap(self, msg):
        # Prefer the sensor stamp; fall back to node time if the driver left
        # it zeroed (Qualisys usually stamps correctly).
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t <= 0.0:
            t = self.get_clock().now().nanoseconds * 1e-9

        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        yaw = quaternion_to_yaw(msg.pose.orientation)

        # Signed longitudinal velocity from position differencing: project the
        # displacement onto the heading, so reverse comes out negative instead
        # of as a positive speed with a flipped yaw.
        if self._t_prev is not None:
            dt = t - self._t_prev
            if 1e-4 < dt < 0.5:
                vx = (x - self._x_prev) / dt
                vy = (y - self._y_prev) / dt
                v = vx * math.cos(yaw) + vy * math.sin(yaw)
                a = float(self.vel_smoothing)
                self._v_ema = a * v + (1.0 - a) * self._v_ema
            elif dt >= 0.5:
                # feed gap: restart the filter rather than smear across it
                self._v_ema = 0.0
        self._t_prev, self._x_prev, self._y_prev = t, x, y

        if self.recording:
            self._write(t, x, y, yaw, self._v_ema)

    # ----------------------------------------------------------------- file IO

    def _open(self):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(
            self.output_dir, f"{self.file_prefix}_{stamp}.csv")
        self._file = open(self._path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["timestamp", "x", "y", "yaw", "velocity", "steering"])
        self._rows = 0
        self._last_write_t = -1.0
        # Old differencing state belongs to a previous run; a stale _t_prev
        # would make the first velocity sample a huge spike.
        self._t_prev = None
        self._v_ema = 0.0
        self.recording = True
        self.get_logger().info(f"recording started -> {self._path}")

    def _close(self, reason):
        self.recording = False
        if self._file is None:
            return
        self._file.flush()
        self._file.close()
        self._file = None
        self._writer = None
        self.get_logger().info(
            f"recording stopped ({reason}): {self._rows} rows in {self._path}")

    def _write(self, t, x, y, yaw, v):
        if self._writer is None:
            return
        if self.min_record_dt > 0.0 and \
                (t - self._last_write_t) < self.min_record_dt:
            return
        self._last_write_t = t
        # Swap `v` for `self._cmd_speed` here if you want the commanded speed
        # in the file instead of the measured one.
        self._writer.writerow([
            f"{t:.6f}",
            f"{x:.6f}",
            f"{y:.6f}",
            f"{yaw:.6f}",
            f"{v:.4f}",
            f"{self._steer:.4f}",
        ])
        self._rows += 1
        # Flush every row: ~100 Hz of short lines is cheap, and a crash or
        # Ctrl-C mid-run must not cost the whole log.
        self._file.flush()

    def on_shutdown(self):
        self._close("shutdown")


if __name__ == "__main__":
    path_recorder.main()
