#! /usr/bin/env python3
"""
Path follower for CSV paths containing cusps (direction reversals).

Place in: csv_path_follower/scripts/path_follower.py

State machine
-------------

    WAIT_ODOM --> DRIVE --> BRAKE --> DRIVE --> ... --> DONE
                    |         |
                    +---------+--> ABORT   (cross-track blown, odom stale)

  WAIT_ODOM  wait for a fresh odometry message and check that the vehicle is
             actually near the start of the path.
  DRIVE      track the current segment with SegmentPurePursuit.
  BRAKE      command zero speed, wait for the wheels to actually stop, pre-steer
             for the next segment, and (if the direction flips) run the ESC
             shift sequence. Only then advance to the next segment.
  DONE       zero everything and stop the timer.
  ABORT      zero everything and complain loudly.

Run:
    bl csv_path_follower path_follower.launch.py is_sim:=True
"""

import math
import os
from enum import Enum, auto

import numpy as np

from svea_core import rosonic as rx
from svea_core.interfaces import (
    ActuationInterface,
    LocalizationInterface,
    ShowMarker,
    ShowPath,
)
from svea_core.controllers.segmented_pure_pursuit import (
    load_segments_from_csv,
    wrap_to_pi,
    SegmentPurePursuit,
)

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Bool

qos_subber = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,  # BEST_EFFORT
    history=QoSHistoryPolicy.KEEP_LAST,         # Keep the last N messages
    durability=QoSDurabilityPolicy.VOLATILE,    # Volatile
    depth=10,                                   # Size of the queue
)


class State(Enum):
    WAIT_ODOM = auto()
    DRIVE = auto()
    BRAKE = auto()
    DONE = auto()
    ABORT = auto()


class path_follower(rx.Node):
    """Follow a CSV path with cusps using segmented pure pursuit."""

    # ---------------------------------------------------------------- params

    path_csv = rx.Parameter("")

    # Geometry. Wheelbase matches PurePursuitController in svea_core.
    wheelbase = rx.Parameter(0.324)

    # 40 deg, matching the limit the path was generated under and the
    # MAX_STEERING_ANGLE that ActuationInterface already assumes.
    #
    # ActuationInterface additionally clips at MAX_STEER_PERCENT = 90, so a
    # 0.698 command actually leaves as 0.628 rad (36 deg). Simulated, that
    # costs almost nothing: cusp heading error 15.8 -> 17.2 deg. Not worth
    # touching the hardware-protection clip for.
    max_steering = rx.Parameter(0.70)

    # steering_gain = (angle ActuationInterface assumes) / (real angle).
    # 1.0 while both are 40 deg. If the mechanical limit turns out to be
    # 30 deg after all, set this to 1.3333 rather than leaving the controller
    # to discover the 0.75 gain error on its own -- simulated, that mismatch
    # pushes the cusp heading error from 15.8 to 19.5 deg at Ld=0.25 (and to
    # 29 deg at Ld=0.35, which is why Ld is 0.25 below).
    steering_gain = rx.Parameter(1.0)

    # Constant servo offset, in radians of commanded angle. Positive steers
    # left. mpc_path_tracking.py carries an equivalent value of 7 unitless
    # counts for svea7 -- convert and put it here if you trust that number.
    steering_bias = rx.Parameter(0.0)

    # The existing pure_pursuit.py example sends velocity * -1.0. Keep -1.0
    # until you have confirmed the sign on the actual car with lli_test.
    velocity_sign = rx.Parameter(-1.0)

    # Speed profile
    target_speed = rx.Parameter(0.35)
    min_speed = rx.Parameter(0.15)
    max_lateral_accel = rx.Parameter(1.2)
    decel_distance = rx.Parameter(0.45)

    # Look-ahead. Defaults are sized for a ~4 m path with ~0.4 m turn radius.
    # The stock PurePursuitController floors this at 1.2 m, which would cut the
    # entire corner off a path this short.
    #
    # Sweep on this path (kinematic sim, 0.35 m/s, 10 Hz, max_steering=0.698),
    # max cross-track / heading error at the cusp / % of steps saturated:
    #
    #   Ld_base    seg0                      seg1 (reverse)
    #    0.40      0.084 m / 22.5 deg /  6%   0.099 m / 13.7 deg
    #    0.35      0.073 m / 20.7 deg /  6%   0.088 m / 11.3 deg
    #    0.30      0.068 m / 16.8 deg / 12%   0.067 m /  7.3 deg
    #    0.25      0.057 m / 15.8 deg /  8%   0.065 m /  5.2 deg   <-- chosen
    #    0.20      0.056 m / 12.1 deg / 20%   0.050 m /  1.8 deg
    #
    # 0.20 tracks marginally better but sits on the steering limit a fifth of
    # the time, which leaves nothing for disturbance rejection. 0.25 keeps
    # saturation at 8% and degrades gracefully: with a 0.20 m / 15 deg initial
    # error the cross-track peaks at 0.200 m and still converges.
    lookahead_base = rx.Parameter(0.25)
    lookahead_gain = rx.Parameter(0.30)
    lookahead_min = rx.Parameter(0.18)
    lookahead_max = rx.Parameter(0.70)
    max_steering_rate = rx.Parameter(3.0)
    steering_margin = rx.Parameter(0.017)

    # Termination / safety
    goal_tolerance = rx.Parameter(0.08)
    max_cross_track = rx.Parameter(0.60)
    start_tolerance = rx.Parameter(1.00)
    odom_timeout = rx.Parameter(0.5)

    # Cusp handling
    stop_speed_threshold = rx.Parameter(0.03)
    brake_dwell = rx.Parameter(0.7)
    brake_timeout = rx.Parameter(4.0)
    # Most RC ESCs in sports mode need neutral -> reverse -> neutral before
    # they will actually accept reverse. Set to 0.0 to disable.
    esc_shift_pulse = rx.Parameter(0.15)

    control_rate = rx.Parameter(10.0)
    difflock = rx.Parameter(True)

    # ------------------------------------------------------------ interfaces

    actuation = ActuationInterface()
    localizer = LocalizationInterface()
    target_marker = ShowMarker()
    path = ShowPath()

    # ---------------------------------------------------------------- startup

    @rx.Subscriber(Bool, '/csv_follower', qos_subber)
    def imu_sub(self, csv_follower_msg):
        self.start = csv_follower_msg.data

    def on_startup(self):
        if not self.path_csv or not os.path.isfile(self.path_csv):
            raise RuntimeError(f"path_csv not found: '{self.path_csv}'")

        self.segments = load_segments_from_csv(self.path_csv,
                                               wheelbase=self.wheelbase)
        self.seg_i = 0

        self.controller = SegmentPurePursuit(
            wheelbase=self.wheelbase,
            lookahead_base=self.lookahead_base,
            lookahead_gain=self.lookahead_gain,
            lookahead_min=self.lookahead_min,
            lookahead_max=self.lookahead_max,
            max_steering=self.max_steering,
            max_steering_rate=self.max_steering_rate,
            target_speed=self.target_speed,
            min_speed=self.min_speed,
            max_lateral_accel=self.max_lateral_accel,
            decel_distance=self.decel_distance,
            goal_tolerance=self.goal_tolerance,
        )

        self._report_path()
        self._publish_full_path()

        self.state = State.WAIT_ODOM
        self.dt = 1.0 / self.control_rate
        self._brake_t = 0.0
        self._stopped_t = None
        self._shift_t = None
        self._last_odom_t = None

        self.start = False

        self.localizer.add_callback(self._on_odom)

        if self.difflock:
            self.actuation.enable_difflock()

        self.create_timer(self.dt, self.loop)
        self.get_logger().info(
            f"path_follower ready: {len(self.segments)} segment(s) from "
            f"{os.path.basename(self.path_csv)}, waiting for odometry")

    def _on_odom(self, msg):
        self._last_odom_t = self.get_clock().now().nanoseconds * 1e-9

    def _report_path(self):
        """Log the segment breakdown and flag anything the car cannot steer."""
        for i, seg in enumerate(self.segments):
            need = seg.max_required_steering()
            tag = "FORWARD" if not seg.is_reverse else "REVERSE"
            msg = (f"segment {i}: {tag}, {len(seg)} pts, {seg.length:.2f} m, "
                   f"max required steering {math.degrees(need):.1f} deg")

            excess = need - self.max_steering

            if excess > self.steering_margin:
                self.get_logger().error(
                    msg + f"  <-- EXCEEDS max_steering by "
                          f"{math.degrees(excess):.1f} deg. The car physically "
                          f"cannot track this segment; it will saturate and "
                          f"drift wide. Regenerate the path with the correct "
                          f"steering limit.")
            elif excess > 0.0:
                self.get_logger().warning(
                    msg + f"  (over by {math.degrees(excess):.2f} deg, "
                          f"within margin -- ignoring)")
            else:
                self.get_logger().info(msg)

    def _publish_full_path(self):
        xs = np.concatenate([s.xs for s in self.segments])
        ys = np.concatenate([s.ys for s in self.segments])
        # NOTE: do not pass yaw_list -- ShowPath.publish_path calls
        # tf.transformations.quaternion_from_euler on a module imported as
        # `import tf_transformations as tf`, which raises AttributeError.
        self.path.publish_path(xs.tolist(), ys.tolist())

    # ------------------------------------------------------------- main loop

    def loop(self):
        if self.start:
            if self.state in (State.DONE, State.ABORT):
                self._halt()
                return

            x, y, yaw, v = self.localizer.get_state()
            now = self.get_clock().now().nanoseconds * 1e-9

            if self._last_odom_t is None:
                self.get_logger().info("waiting for odometry...",
                                    throttle_duration_sec=2.0)
                self._halt()
                return

            if self.state is not State.WAIT_ODOM and \
                    (now - self._last_odom_t) > self.odom_timeout:
                return self._abort(
                    f"odometry stale by {now - self._last_odom_t:.2f} s")

            handler = {
                State.WAIT_ODOM: self._do_wait,
                State.DRIVE: self._do_drive,
                State.BRAKE: self._do_brake,
            }[self.state]
            handler(x, y, yaw, v, now)
        else:
            self._halt()


    # ----------------------------------------------------------- state: WAIT

    def _do_wait(self, x, y, yaw, v, now):
        self._halt()

        seg = self.segments[0]
        d = math.hypot(x - seg.xs[0], y - seg.ys[0])
        if d > self.start_tolerance:
            self.get_logger().warning(
                f"vehicle is {d:.2f} m from the path start "
                f"({seg.xs[0]:.2f}, {seg.ys[0]:.2f}) -- "
                f"check the initial pose before it drives off",
                throttle_duration_sec=3.0)
            return

        yaw_err = abs(wrap_to_pi(yaw - seg.yaws[0]))
        if yaw_err > math.radians(45):
            self.get_logger().warning(
                f"heading is {math.degrees(yaw_err):.0f} deg off the path "
                f"start heading -- pure pursuit will swing wide",
                throttle_duration_sec=3.0)

        self.controller.set_segment(seg)
        self.state = State.DRIVE
        self.get_logger().info(f"--> DRIVE segment 0 "
                               f"({'reverse' if seg.is_reverse else 'forward'})")

    # ---------------------------------------------------------- state: DRIVE

    def _do_drive(self, x, y, yaw, v, now):
        delta, speed, info = self.controller.compute(x, y, yaw, v, self.dt)

        if info["cross_track"] > self.max_cross_track:
            return self._abort(
                f"cross-track error {info['cross_track']:.2f} m > "
                f"{self.max_cross_track:.2f} m on segment {self.seg_i}")

        self._send(delta, speed)

        tx, ty = info["target"]
        self.target_marker.place([tx, ty, 0.2], color="lime", scale=0.15)

        self.get_logger().info(
            f"seg {self.seg_i} i={info['index']}/{len(self.controller.seg)-1} "
            f"e={info['cross_track']:.3f} rem={info['remaining']:.2f} "
            f"Ld={info['lookahead']:.2f} d={math.degrees(delta):+.1f} "
            f"v={speed:+.2f}",
            throttle_duration_sec=0.5)

        if info["finished"]:
            self.state = State.BRAKE
            self._brake_t = now
            self._stopped_t = None
            self._shift_t = None
            self.get_logger().info(
                f"--> BRAKE (segment {self.seg_i} done, "
                f"final cross-track {info['cross_track']:.3f} m)")

    # ---------------------------------------------------------- state: BRAKE

    def _do_brake(self, x, y, yaw, v, now):
        nxt = self.seg_i + 1
        has_next = nxt < len(self.segments)

        # Hold the steering the next segment wants, so the servo is already in
        # position when we start moving. Straight if this is the last stop.
        pre_steer = 0.0
        if has_next:
            s_next = self.segments[nxt]
            pre_steer = math.atan(self.wheelbase * float(s_next.curvature[0]))
            if s_next.is_reverse:
                pre_steer = -pre_steer
            pre_steer = float(np.clip(pre_steer,
                                      -self.max_steering, self.max_steering))

        # 1. wait for the wheels to actually stop
        if self._stopped_t is None:
            self._send(pre_steer, 0.0)
            if abs(v) < self.stop_speed_threshold:
                self._stopped_t = now
            elif (now - self._brake_t) > self.brake_timeout:
                self.get_logger().warning(
                    f"still reading |v|={abs(v):.2f} m/s after "
                    f"{self.brake_timeout:.1f} s -- assuming stopped. If the "
                    f"car is visibly still, the encoder/twist filter is lying.")
                self._stopped_t = now
            return

        if (now - self._stopped_t) < self.brake_dwell:
            self._send(pre_steer, 0.0)
            return

        if not has_next:
            self.get_logger().info("--> DONE (all segments complete)")
            self.state = State.DONE
            self._halt()
            return

        # 2. ESC shift sequence, only when the direction actually flips
        flips = self.segments[nxt].direction != self.segments[self.seg_i].direction
        if flips and self.esc_shift_pulse > 0.0:
            if self._shift_t is None:
                self._shift_t = now
            elapsed = now - self._shift_t
            if elapsed < self.esc_shift_pulse:
                # brief pulse in the new direction to unlock the ESC
                self._send(pre_steer,
                           0.3 * self.min_speed * self.segments[nxt].direction)
                return
            if elapsed < 2.0 * self.esc_shift_pulse:
                self._send(pre_steer, 0.0)  # back to neutral
                return

        # 3. advance
        self.seg_i = nxt
        seg = self.segments[self.seg_i]
        self.controller.set_segment(seg)
        self.state = State.DRIVE
        self.get_logger().info(
            f"--> DRIVE segment {self.seg_i} "
            f"({'reverse' if seg.is_reverse else 'forward'}, "
            f"{seg.length:.2f} m)")

    # ------------------------------------------------------------- actuation

    def _send(self, delta, speed):
        """Apply the calibration corrections and push to the LLI."""
        delta = float(np.clip(delta, -self.max_steering, self.max_steering))
        cmd_steer = delta * self.steering_gain + self.steering_bias
        cmd_speed = speed * self.velocity_sign
        self.actuation.send_control(cmd_steer, -cmd_speed)

    def _halt(self):
        self.actuation.send_control(0.0, 0.0)

    def _abort(self, reason):
        self.get_logger().error(f"ABORT: {reason}")
        self.state = State.ABORT
        self._halt()

    def on_shutdown(self):
        self._halt()


if __name__ == "__main__":
    path_follower.main()