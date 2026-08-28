#! /usr/bin/env python3
"""
Path follower for CSV paths containing cusps (direction reversals).

Place in: csv_path_follower/scripts/path_follower.py

Frames
------

The CSV and the mocap topic are both expressed in path_frame; everything the
controller does happens in target_frame:

    path_frame     the frame the CSV and the mocap pose are in  (e.g. 'mocap')
    target_frame   the frame everything is converted into       (e.g. 'map')

When they differ the node waits for a planar transform target_frame <-
path_frame, caches it, rewrites the CSV with it, and applies the SAME cached
transform to every incoming mocap pose. Path, vehicle pose, target markers and
map therefore all live in one frame and cannot drift apart.

Converting only the path and leaving the pose in the mocap frame (or the other
way round) yields a pure pursuit bearing that is wrong by the rotation between
the frames. Nothing errors; the car simply drives somewhere else.

Pose sources
------------

    is_sim=True    pose and speed both from LocalizationInterface.
    is_sim=False   pose from mocap_pose_topic (converted), speed still from
                   LocalizationInterface (wheel encoders).

Both feeds are staleness-checked independently: a dead mocap feed alongside a
live encoder feed would otherwise leave the controller steering against a
frozen pose.

State machine
-------------

    WAIT_TF --> WAIT_ODOM --> DRIVE --> BRAKE --> DRIVE --> ... --> DONE
                                 |        |
                                 +--------+--> ABORT  (cross-track, stale pose)

Run:
    bl csv_path_follower path_follower.launch.py is_sim:=True
"""

import csv
import math
import os
import tempfile
from enum import Enum, auto

import numpy as np

import rclpy
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistStamped

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

qos_subber = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=10,
)

# initialpose is a one-shot seed. Latch it so a consumer that starts late
# still receives it instead of missing it entirely.
qos_latched = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    depth=1,
)


def quaternion_to_yaw(q) -> float:
    """Planar yaw from a quaternion. Ignores roll/pitch by construction."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def apply_planar_tf(tf, x, y, yaw):
    """Apply (tx, ty, theta) to a planar pose. tf=None is the identity.

    One function for both the CSV rows and the live mocap poses, so the two
    conversions cannot diverge.
    """
    if tf is None:
        return x, y, yaw
    tx, ty, th = tf
    c, s = math.cos(th), math.sin(th)
    return tx + c * x - s * y, ty + s * x + c * y, wrap_to_pi(yaw + th)


class State(Enum):
    WAIT_TF = auto()
    WAIT_ODOM = auto()
    DRIVE = auto()
    BRAKE = auto()
    DONE = auto()
    ABORT = auto()


class path_follower(rx.Node):
    """Follow a CSV path with cusps using segmented pure pursuit."""

    # ---------------------------------------------------------------- params

    path_csv = rx.Parameter("")

    # --- frames ---------------------------------------------------------
    # path_frame:   what the CSV numbers AND the mocap topic are expressed in.
    # target_frame: what everything is converted into. Set the two equal (or
    #               leave target_frame empty) to disable conversion entirely.
    path_frame = rx.Parameter("mocap")
    target_frame = rx.Parameter("map")

    # How long to wait for the transform before aborting. The vehicle is held
    # at zero throttle for the whole wait, so a generous value is safe.
    tf_timeout = rx.Parameter(15.0)

    # Hand-supplied planar extrinsic target_frame <- path_frame, used when
    # use_tf_override is true. Three scalars rather than a list: rosonic's
    # declarative parameters and rcl both handle scalars more predictably,
    # and it sidesteps the PyYAML float-notation traps.
    use_tf_override = rx.Parameter(False)
    tf_override_x = rx.Parameter(0.0)
    tf_override_y = rx.Parameter(0.0)
    tf_override_yaw = rx.Parameter(0.0)

    # --- mocap ------------------------------------------------------------
    is_sim = rx.Parameter(False)

    mocap_pose_topic = rx.Parameter("/svea7/pose")
    mocap_timeout = rx.Parameter(0.5)

    # Seed the localizer with the first converted mocap pose. Set false when
    # something else already owns initialpose -- two seeds fighting is worse
    # than none.
    publish_initial_pose = rx.Parameter(True)
    # Relative by default so bl.group(name) namespaces it. A leading slash
    # pins it to the root and a localizer under /<name> never hears it.
    initial_pose_topic = rx.Parameter("/self/set_pose")

    # --- geometry ---------------------------------------------------------
    wheelbase = rx.Parameter(0.324)

    # 40 deg, matching the limit the path was generated under and the
    # MAX_STEERING_ANGLE that ActuationInterface already assumes.
    max_steering = rx.Parameter(0.70)

    # steering_gain = (angle ActuationInterface assumes) / (real angle).
    steering_gain = rx.Parameter(1.0)

    # Constant servo offset, in radians of commanded angle. Positive steers
    # left.
    steering_bias = rx.Parameter(0.0)

    # Sign applied to the commanded velocity before it reaches the LLI.
    velocity_sign = rx.Parameter(-1.0)

    # --- speed profile ----------------------------------------------------
    target_speed = rx.Parameter(0.35)
    min_speed = rx.Parameter(0.30)
    max_lateral_accel = rx.Parameter(1.2)
    decel_distance = rx.Parameter(0.45)

    # --- look-ahead -------------------------------------------------------
    # Sweep on this path (kinematic sim, 0.35 m/s, 10 Hz, max_steering=0.698),
    # max cross-track / heading error at the cusp / % of steps saturated:
    #
    #   Ld_base    seg0                      seg1 (reverse)
    #    0.40      0.084 m / 22.5 deg /  6%   0.099 m / 13.7 deg
    #    0.35      0.073 m / 20.7 deg /  6%   0.088 m / 11.3 deg
    #    0.30      0.068 m / 16.8 deg / 12%   0.067 m /  7.3 deg
    #    0.25      0.057 m / 15.8 deg /  8%   0.065 m /  5.2 deg   <-- chosen
    #    0.20      0.056 m / 12.1 deg / 20%   0.050 m /  1.8 deg
    lookahead_base = rx.Parameter(0.25)
    lookahead_gain = rx.Parameter(0.30)
    lookahead_min = rx.Parameter(0.18)
    lookahead_max = rx.Parameter(0.70)
    max_steering_rate = rx.Parameter(3.0)
    steering_margin = rx.Parameter(0.017)

    # --- termination / safety --------------------------------------------
    goal_tolerance = rx.Parameter(0.08)
    max_cross_track = rx.Parameter(0.60)
    start_tolerance = rx.Parameter(1.00)
    odom_timeout = rx.Parameter(0.5)

    # --- cusp handling ----------------------------------------------------
    stop_speed_threshold = rx.Parameter(0.03)
    brake_dwell = rx.Parameter(0.7)
    brake_timeout = rx.Parameter(4.0)
    esc_shift_pulse = rx.Parameter(0.15)

    control_rate = rx.Parameter(10.0)
    difflock = rx.Parameter(True)

    # Hold the throttle for this long after seeding, so the filter has a cycle
    # or two to converge onto the new pose before the car starts moving.
    initial_pose_settle = rx.Parameter(1.0)

    # ------------------------------------------------------------ interfaces

    actuation = ActuationInterface()
    localizer = LocalizationInterface()
    target_marker = ShowMarker()
    path = ShowPath()

    # ---------------------------------------------------------------- startup

    @rx.Subscriber(Bool, '/csv_follower', qos_subber)
    def start_sub(self, msg):
        if not msg.data:
            self.start = False
            self._settle_until = None
            return
        # Seed first, then let the filter settle before releasing the throttle.
        # The delay is handled in loop() -- sleeping here would block the
        # executor, stalling the control timer and the halt commands with it.
        if not self._initial_pose_sent and self._pose_is_converted():
            self._send_initial_pose()
            self._settle_until = (self.get_clock().now().nanoseconds * 1e-9
                                  + self.initial_pose_settle)
        self.start = True

    # NOTE: the mocap subscription and the initialpose publisher are created in
    # on_startup, not declared here. Decorator and rx.Publisher arguments are
    # evaluated at class-definition time, when mocap_pose_topic is still an
    # rx.Parameter descriptor rather than the configured string -- the node
    # would come up subscribed to the wrong name, silently.

    def on_startup(self):
        # Set before anything can arrive, so an early message cannot hit an
        # undefined attribute.
        self.start = False
        self.x = self.y = self.yaw = 0.0
        self._tf = None
        self._initial_pose_sent = False
        self._last_mocap_t = None
        self._last_odom_t = None

        if not self.path_csv or not os.path.isfile(self.path_csv):
            raise RuntimeError(f"path_csv not found: '{self.path_csv}'")

        self.segments = None
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

        self.dt = 1.0 / self.control_rate
        self._brake_t = 0.0
        self._stopped_t = None
        self._shift_t = None
        self._tf_t0 = None
        self._tf_buffer = None
        self._tf_listener = None

        self.localizer.add_callback(self._on_odom)

        # --- mocap plumbing, hardware only -------------------------------
        self._initial_pose_pub = None
        if not self.is_sim:
            if self.publish_initial_pose:
                self._initial_pose_pub = self.create_publisher(
                    PoseWithCovarianceStamped,
                    self.initial_pose_topic, qos_latched)
            self.create_subscription(
                PoseStamped, self.mocap_pose_topic,
                self._on_mocap_pose, qos_subber)
            self.get_logger().info(
                f"pose from '{self.mocap_pose_topic}' (frame "
                f"'{self.path_frame}'), initial pose to "
                f"'{self.initial_pose_topic}'"
                if self.publish_initial_pose else
                f"pose from '{self.mocap_pose_topic}' (frame "
                f"'{self.path_frame}'), initial pose disabled")
        else:
            self.get_logger().info("pose from LocalizationInterface (sim)")

        # --- recording hooks ----------------------------------------------
        # Relative names on purpose: bl.group(name) namespaces them, so the
        # recorder launched in the same group hears them without any leading
        # slashes (leading slashes would double-nest, see launch notes).
        #   ctrl_info  TwistStamped: angular.z = steering [rad] (pre gain/bias),
        #              linear.x = signed commanded speed [m/s]
        #   path_done  Bool, latched: True once the last segment completes or
        #              the follower aborts.
        self._ctrl_info_pub = self.create_publisher(
            TwistStamped, "ctrl_info", 10)
        self._done_pub = self.create_publisher(
            Bool, "path_done", qos_latched)

        if self.difflock:
            self.actuation.enable_difflock()

        # --- decide whether a frame conversion is needed -----------------
        needs_tf = bool(self.target_frame) and self.target_frame != self.path_frame

        if not needs_tf:
            self.get_logger().info(
                f"everything stays in '{self.path_frame}'; no frame conversion")
            self._load_path(None)
            self.state = State.WAIT_ODOM
        elif self.use_tf_override:
            self._apply_transform(
                (float(self.tf_override_x),
                 float(self.tf_override_y),
                 float(self.tf_override_yaw)),
                "tf_override parameters")
            self.state = State.WAIT_ODOM
        else:
            from tf2_ros import Buffer, TransformListener
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
            self.state = State.WAIT_TF
            self.get_logger().info(
                f"waiting for transform {self.target_frame} <- "
                f"{self.path_frame} before loading the path")

        self.create_timer(self.dt, self.loop)
        self.get_logger().info("path_follower ready")

    # -------------------------------------------------------------- callbacks

    def _on_odom(self, msg):
        self._last_odom_t = self.get_clock().now().nanoseconds * 1e-9

    def _on_mocap_pose(self, msg):
        """Receive a mocap pose, convert it into target_frame, stash it.

        Uses the transform cached by _apply_transform, so the pose and the path
        are guaranteed to have gone through the same conversion.

        Before the transform resolves, self._tf is None and apply_planar_tf is
        the identity. That is harmless: the state machine is still in WAIT_TF,
        the car is held at zero throttle, and no initialpose is emitted until
        the transform exists.
        """
        x = msg.pose.position.x
        y = msg.pose.position.y
        yaw = quaternion_to_yaw(msg.pose.orientation)

        self.x, self.y, self.yaw = apply_planar_tf(self._tf, x, y, yaw)
        self._last_mocap_t = self.get_clock().now().nanoseconds * 1e-9


    def _pose_is_converted(self):
        """True once self.x/y/yaw are genuinely in target_frame."""
        return self._tf is not None or not self._needs_conversion()

    def _needs_conversion(self):
        return bool(self.target_frame) and self.target_frame != self.path_frame

    def _send_initial_pose(self):
        """Seed the localizer with the converted mocap pose.

        Three things this must get right, all of which have bitten before:

        - PoseWithCovarianceStamped, not PoseStamped. That is what nav2_amcl
          and robot_localization accept on this topic.
        - header.frame_id must name the target frame. An empty string is
          rejected outright, and publishing the raw mocap frame relies on the
          consumer doing a TF lookup that it may not do -- some versions read
          the numbers as if they were already in the map frame, which puts the
          seed off by the whole mocap->map offset with no error anywhere.
        - The covariance must not be all zeros, or every particle stacks onto
          a single point and the filter cannot recover.
        """
        if self._initial_pose_pub is None:
            return

        out = PoseWithCovarianceStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.target_frame or self.path_frame
        out.pose.pose.position.x = float(self.x)
        out.pose.pose.position.y = float(self.y)
        out.pose.pose.position.z = 0.0
        out.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        out.pose.pose.orientation.w = math.cos(self.yaw / 2.0)

        cov = [0.0] * 36
        cov[0] = 0.25       # x
        cov[7] = 0.25       # y
        cov[35] = 0.0685    # yaw
        out.pose.covariance = cov

        self._initial_pose_pub.publish(out)
        self._initial_pose_sent = True
        self.get_logger().info(
            f"published initial pose ({self.x:+.3f}, {self.y:+.3f}, "
            f"{math.degrees(self.yaw):+.1f} deg) in "
            f"'{out.header.frame_id}' on '{self.initial_pose_topic}'")

    # ------------------------------------------------------------ path load

    def _load_path(self, csv_override):
        """Build the segments, from the original CSV or a rewritten one."""
        source = csv_override or self.path_csv
        self.segments = load_segments_from_csv(source, wheelbase=self.wheelbase)
        self.seg_i = 0
        self._report_path()
        self._publish_full_path()

    def _apply_transform(self, tf, source):
        """Cache the transform, rewrite the CSV into target_frame, then load it.

        Rewriting the file rather than mutating the loaded segment arrays means
        every derived quantity -- curvature, arc length, direction, and anything
        else PathSegment caches -- is computed from the transformed coordinates.
        A planar rigid transform leaves curvature and length unchanged, so both
        routes agree mathematically, but this one cannot go stale if
        segmented_pure_pursuit.py grows a new cached field later.
        """
        self._tf = tf
        tx, ty, th = tf

        rows = []
        with open(self.path_csv, "r") as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames or [])
            missing = {"x", "y", "yaw"} - set(fields)
            if missing:
                raise RuntimeError(
                    f"CSV is missing column(s) {sorted(missing)}. Found: {fields}")
            for row in reader:
                x, y, yaw = apply_planar_tf(
                    tf, float(row["x"]), float(row["y"]), float(row["yaw"]))
                row["x"] = f"{x:.6f}"
                row["y"] = f"{y:.6f}"
                row["yaw"] = f"{yaw:.6f}"
                rows.append(row)

        if not rows:
            raise RuntimeError(f"no data rows in '{self.path_csv}'")

        base = os.path.basename(self.path_csv).rsplit(".", 1)[0]
        out = os.path.join(tempfile.gettempdir(),
                           f"{base}.{self.target_frame.replace('/', '_')}.csv")
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        self.get_logger().info(
            f"transform {self.target_frame} <- {self.path_frame} from {source}: "
            f"translation ({tx:+.3f}, {ty:+.3f}) m, rotation "
            f"{math.degrees(th):+.2f} deg")
        self.get_logger().info(f"  rewrote {len(rows)} rows into {out}")

        self._load_path(out)

        # A mocap pose may already have arrived and been stored unconverted.
        # Re-run the conversion on the raw values is not possible (they were
        # overwritten), so simply wait for the next message -- at mocap rates
        # that is a few milliseconds. Emit the seed from that one instead.
        self._last_mocap_t = None

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
        if self.state in (State.DONE, State.ABORT):
            self._halt()
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        # Resolve the transform regardless of self.start. It costs nothing,
        # the car is held at zero throttle throughout, and having it done
        # early means the start signal is not delayed by a TF handshake.
        if self.state is State.WAIT_TF:
            self._do_wait_tf(now)
            return

        if not self.start:
            self._halt()
            return

        if self._settle_until is not None:
            if now < self._settle_until:
                self._halt()
                self.get_logger().info(
                    "waiting for the localizer to settle after the pose seed...",
                    throttle_duration_sec=0.5)
                return
            self._settle_until = None
            self.get_logger().info("localizer settled, releasing throttle")

        # Speed always from the localizer (wheel encoders). Pose from the
        # localizer in simulation, from mocap on hardware.
        lx, ly, lyaw, v = self.localizer.get_state()
        if self.is_sim:
            x, y, yaw = lx, ly, lyaw
        else:
            x, y, yaw = self.x, self.y, self.yaw

        if not self._pose_ready():
            self._halt()
            return

        if self.state is not State.WAIT_ODOM and not self._pose_fresh(now):
            return

        handler = {
            State.WAIT_ODOM: self._do_wait,
            State.DRIVE: self._do_drive,
            State.BRAKE: self._do_brake,
        }[self.state]
        handler(x, y, yaw, v, now)

    def _pose_ready(self):
        if self._last_odom_t is None:
            self.get_logger().info("waiting for odometry...",
                                   throttle_duration_sec=2.0)
            return False
        if not self.is_sim and self._last_mocap_t is None:
            self.get_logger().info(
                f"waiting for mocap pose on '{self.mocap_pose_topic}'...",
                throttle_duration_sec=2.0)
            return False
        return True

    def _pose_fresh(self, now):
        """Check both feeds. A live encoder feed alongside a dead mocap feed
        would otherwise leave the controller steering against a frozen pose --
        the pure pursuit target never appears to be reached, so the steering
        winds up and stays there."""
        age = now - self._last_odom_t
        if age > self.odom_timeout:
            self._abort(f"odometry stale by {age:.2f} s")
            return False
        if not self.is_sim:
            age = now - self._last_mocap_t
            if age > self.mocap_timeout:
                self._abort(f"mocap pose stale by {age:.2f} s on "
                            f"'{self.mocap_pose_topic}'")
                return False
        return True

    # ------------------------------------------------------------- state: TF

    def _do_wait_tf(self, now):
        self._halt()

        if self._tf_t0 is None:
            self._tf_t0 = now

        from tf2_ros import TransformException
        try:
            tr = self._tf_buffer.lookup_transform(
                self.target_frame, self.path_frame, rclpy.time.Time()).transform
        except TransformException as exc:
            if (now - self._tf_t0) > self.tf_timeout:
                return self._abort(
                    f"no transform {self.target_frame} <- {self.path_frame} "
                    f"after {self.tf_timeout:.0f} s: {exc}. Either the static "
                    f"broadcaster is not running, or set use_tf_override with "
                    f"the extrinsic measured by hand.")
            self.get_logger().info(
                f"waiting for {self.target_frame} <- {self.path_frame}...",
                throttle_duration_sec=2.0)
            return

        if abs(tr.translation.z) > 1e-3:
            self.get_logger().warning(
                f"transform has z offset {tr.translation.z:+.3f} m; this node "
                f"is planar and drops it")
        q = tr.rotation
        if abs(q.x) > 1e-3 or abs(q.y) > 1e-3:
            self.get_logger().warning(
                "transform has non-negligible roll/pitch; only yaw is kept, "
                "so the converted path will be approximate")

        self._apply_transform(
            (tr.translation.x, tr.translation.y, quaternion_to_yaw(q)), "TF")
        self.state = State.WAIT_ODOM

    # ----------------------------------------------------------- state: WAIT

    def _do_wait(self, x, y, yaw, v, now):
        self._halt()

        seg = self.segments[0]
        d = math.hypot(x - seg.xs[0], y - seg.ys[0])
        if d > self.start_tolerance:
            self.get_logger().warning(
                f"vehicle is at ({x:+.2f}, {y:+.2f}), {d:.2f} m from the path "
                f"start ({seg.xs[0]:+.2f}, {seg.ys[0]:+.2f}) in "
                f"'{self.target_frame}' -- check the initial pose before it "
                f"drives off",
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
            self._publish_done()
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
        if self.is_sim:
            self.actuation.send_control(cmd_steer, -cmd_speed)
        else:
            self.actuation.send_control(cmd_steer, cmd_speed)
        self._publish_ctrl_info(delta, speed)

    def _halt(self):
        self.actuation.send_control(0.0, 0.0)
        self._publish_ctrl_info(0.0, 0.0)

    def _publish_ctrl_info(self, delta, speed):
        """Mirror the logical command (steering angle, signed speed) for the
        path recorder. Logical, not calibrated: gain/bias/velocity_sign are
        servo-side corrections and would only obscure the recorded data."""
        pub = getattr(self, "_ctrl_info_pub", None)
        if pub is None:
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.angular.z = float(delta)
        msg.twist.linear.x = float(speed)
        pub.publish(msg)

    def _publish_done(self):
        pub = getattr(self, "_done_pub", None)
        if pub is not None:
            pub.publish(Bool(data=True))

    def _abort(self, reason):
        self.get_logger().error(f"ABORT: {reason}")
        self.state = State.ABORT
        self._halt()
        self._publish_done()

    def on_shutdown(self):
        self._halt()


if __name__ == "__main__":
    path_follower.main()