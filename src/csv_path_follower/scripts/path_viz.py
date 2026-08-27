#! /usr/bin/env python3
"""
Visualise a CSV path in the map frame. Nothing else -- no control, no actuation.

Deliberately depends on rclpy and standard message packages only: no rosonic,
no svea_core, no tf_transformations. That means it runs straight from the
source tree without colcon build and without an executable bit:

    python3 src/csv_path_follower/scripts/path_viz.py --ros-args \
        -p path_csv:=/svea_ws/src/csv_path_follower/paths/tiha_path_data.csv \
        -p map_yaml:=/svea_ws/src/svea_core/maps/tum_d_floor3.yaml

Expected CSV columns: x, y, yaw, direction.

Published topics (all latched via transient_local, and republished on a timer,
so Foxglove picks them up whenever it connects -- the transient_local handshake
across machines can take a few seconds):

    ~/path            nav_msgs/Path             every point, ordered
    ~/poses_forward   geometry_msgs/PoseArray   heading arrows, direction=+1
    ~/poses_reverse   geometry_msgs/PoseArray   heading arrows, direction=-1
    ~/markers         visualization_msgs/MarkerArray
                          coloured polyline per segment, point dots,
                          cusp spheres, start/end/cusp text labels
"""

import csv
import math
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)

from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

FORWARD_RGB = (0.20, 0.60, 1.00)
REVERSE_RGB = (1.00, 0.55, 0.10)
CUSP_RGB = (1.00, 0.15, 0.15)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """Rotation about z only. Avoids depending on tf_transformations."""
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def rgba(rgb, a=1.0) -> ColorRGBA:
    return ColorRGBA(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), a=float(a))


class PathViz(Node):

    def __init__(self):
        super().__init__("path_viz")

        self.declare_parameter("path_csv", "/svea_ws/src/csv_path_follower/paths/tiha_path_data.csv")
        self.declare_parameter("map_yaml", "/svea_ws/src/svea_core/maps/sml.yaml")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_period", 1.0)
        self.declare_parameter("point_size", 0.04)
        self.declare_parameter("line_width", 0.02)
        self.declare_parameter("arrow_length", 0.12)
        self.declare_parameter("label_size", 0.15)

        self.path_csv = self.get_parameter("path_csv").value
        self.map_yaml = self.get_parameter("map_yaml").value
        self.frame_id = self.get_parameter("frame_id").value
        self.point_size = float(self.get_parameter("point_size").value)
        self.line_width = float(self.get_parameter("line_width").value)
        self.arrow_length = float(self.get_parameter("arrow_length").value)
        self.label_size = float(self.get_parameter("label_size").value)

        if not self.path_csv:
            raise RuntimeError(
                "path_csv is empty. Pass it: --ros-args -p path_csv:=/abs/path.csv")
        if not os.path.isfile(self.path_csv):
            raise RuntimeError(f"path_csv does not exist: '{self.path_csv}'")

        self.xs, self.ys, self.yaws, self.dirs = self._read_csv(self.path_csv)
        self.segments = self._split_segments()

        self._report_geometry()
        if self.map_yaml:
            self._report_map_fit()

        # Latch everything. Depth must be >= the number of messages we want
        # retained per topic; 1 is enough since each topic carries one message.
        qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub_path = self.create_publisher(Path, "~/path", qos)
        self.pub_fwd = self.create_publisher(PoseArray, "~/poses_forward", qos)
        self.pub_rev = self.create_publisher(PoseArray, "~/poses_reverse", qos)
        self.pub_markers = self.create_publisher(MarkerArray, "~/markers", qos)

        self._build_messages()
        self._publish()

        period = float(self.get_parameter("publish_period").value)
        if period > 0.0:
            self.create_timer(period, self._publish)

        ns = self.get_namespace().rstrip("/")
        self.get_logger().info(
            f"publishing on {ns}/path_viz/path, /poses_forward, /poses_reverse, "
            f"/markers  (frame '{self.frame_id}')")

    # ------------------------------------------------------------------ read

    def _read_csv(self, filename):
        xs, ys, yaws, dirs = [], [], [], []
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            missing = {"x", "y", "yaw", "direction"} - set(reader.fieldnames or [])
            if missing:
                raise RuntimeError(
                    f"CSV is missing column(s) {sorted(missing)}. "
                    f"Found: {reader.fieldnames}")
            for n, row in enumerate(reader):
                try:
                    xs.append(float(row["x"]))
                    ys.append(float(row["y"]))
                    yaws.append(float(row["yaw"]))
                    dirs.append(int(round(float(row["direction"]))))
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(f"bad value on data row {n}: {row}") from exc
        if len(xs) < 2:
            raise RuntimeError(f"only {len(xs)} point(s) in '{filename}'")
        return xs, ys, yaws, dirs

    def _split_segments(self):
        """[(start_index, end_index_inclusive, direction), ...]"""
        segs = []
        start = 0
        for i in range(1, len(self.dirs) + 1):
            if i == len(self.dirs) or self.dirs[i] != self.dirs[start]:
                segs.append((start, i - 1, self.dirs[start]))
                start = i
        return segs

    # ---------------------------------------------------------------- report

    def _report_geometry(self):
        xs, ys, yaws = self.xs, self.ys, self.yaws
        ds = [math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
              for i in range(len(xs) - 1)]
        total = sum(ds)

        log = self.get_logger()
        log.info(f"loaded {len(xs)} points from {os.path.basename(self.path_csv)}")
        log.info(f"  extent   x [{min(xs):+.2f}, {max(xs):+.2f}]  "
                 f"y [{min(ys):+.2f}, {max(ys):+.2f}]")
        log.info(f"  length   {total:.3f} m, spacing "
                 f"min {min(ds):.3f} / mean {total/len(ds):.3f} / max {max(ds):.3f} m")
        log.info(f"  start    ({xs[0]:+.3f}, {ys[0]:+.3f}) "
                 f"yaw {math.degrees(yaws[0]):+.1f} deg")
        log.info(f"  end      ({xs[-1]:+.3f}, {ys[-1]:+.3f}) "
                 f"yaw {math.degrees(yaws[-1]):+.1f} deg")

        if max(abs(y) for y in yaws) > 7.0:
            log.warning("  yaw values exceed +-7 rad -- is this column in degrees? "
                        "This node assumes radians.")

        for i, (a, b, d) in enumerate(self.segments):
            seg_len = sum(ds[a:b]) if b > a else 0.0
            log.info(f"  segment {i}: {'FORWARD' if d > 0 else 'REVERSE'}, "
                     f"rows {a}..{b} ({b - a + 1} pts), {seg_len:.3f} m")

        for i in range(len(self.segments) - 1):
            j = self.segments[i][1]
            gap = math.hypot(xs[j + 1] - xs[j], ys[j + 1] - ys[j])
            dyaw = math.degrees(abs((yaws[j + 1] - yaws[j] + math.pi)
                                    % (2 * math.pi) - math.pi))
            log.info(f"  cusp after row {j} at ({xs[j]:+.3f}, {ys[j]:+.3f}): "
                     f"gap {gap:.3f} m, heading change {dyaw:.1f} deg")

    def _report_map_fit(self):
        """Compare the path extent against the map, so a frame or origin
        mismatch shows up as numbers rather than as a puzzling picture."""
        log = self.get_logger()
        if not os.path.isfile(self.map_yaml):
            log.warning(f"map_yaml does not exist: '{self.map_yaml}'")
            return
        try:
            import yaml
            with open(self.map_yaml) as f:
                meta = yaml.safe_load(f)
            res = float(meta["resolution"])
            ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
            img = meta["image"]
            if not os.path.isabs(img):
                img = os.path.join(os.path.dirname(self.map_yaml), img)
            width, height = self._image_size(img)
        except Exception as exc:  # noqa: BLE001 -- diagnostics only
            log.warning(f"could not read map metadata: {exc}")
            return

        xmin, xmax = ox, ox + width * res
        ymin, ymax = oy, oy + height * res
        log.info(f"map '{os.path.basename(self.map_yaml)}': {width}x{height} px @ "
                 f"{res} m/px  ->  x [{xmin:+.2f}, {xmax:+.2f}] "
                 f"y [{ymin:+.2f}, {ymax:+.2f}]")

        outside = sum(1 for x, y in zip(self.xs, self.ys)
                      if not (xmin <= x <= xmax and ymin <= y <= ymax))
        if outside:
            log.error(f"{outside}/{len(self.xs)} path points fall OUTSIDE the map. "
                      f"Either the path was generated in a different frame, or "
                      f"against a different map, or the origin in the map yaml "
                      f"has changed since.")
        else:
            log.info("all path points lie inside the map bounds")

    @staticmethod
    def _image_size(filename):
        """Size of a PGM/PNG without requiring PIL."""
        try:
            from PIL import Image
            with Image.open(filename) as im:
                return im.size
        except ImportError:
            pass
        # Minimal binary PGM (P5) header parser.
        with open(filename, "rb") as f:
            fields = []
            magic = f.readline().strip()
            if magic not in (b"P5", b"P2"):
                raise RuntimeError(f"install Pillow to read {magic!r} images")
            while len(fields) < 2:
                line = f.readline()
                if not line:
                    raise RuntimeError("truncated PGM header")
                if line.startswith(b"#"):
                    continue
                fields.extend(line.split())
            return int(fields[0]), int(fields[1])

    # ----------------------------------------------------------------- build

    def _stamp_header(self, msg):
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()

    def _build_messages(self):
        # --- Path ------------------------------------------------------
        path = Path()
        for x, y, yaw in zip(self.xs, self.ys, self.yaws):
            ps = PoseStamped()
            ps.header.frame_id = self.frame_id
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation = yaw_to_quaternion(yaw)
            path.poses.append(ps)
        self.msg_path = path

        # --- PoseArrays, split by direction ---------------------------
        fwd, rev = PoseArray(), PoseArray()
        for x, y, yaw, d in zip(self.xs, self.ys, self.yaws, self.dirs):
            p = Pose()
            p.position.x = x
            p.position.y = y
            p.orientation = yaw_to_quaternion(yaw)
            (fwd if d > 0 else rev).poses.append(p)
        self.msg_fwd, self.msg_rev = fwd, rev

        # --- MarkerArray ----------------------------------------------
        markers = MarkerArray()
        mid = 0

        for i, (a, b, d) in enumerate(self.segments):
            colour = FORWARD_RGB if d > 0 else REVERSE_RGB

            line = Marker()
            line.ns = "segment"
            line.id = mid
            mid += 1
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = self.line_width
            line.color = rgba(colour, 0.95)
            line.pose.orientation.w = 1.0
            line.points = [Point(x=self.xs[k], y=self.ys[k], z=0.0)
                           for k in range(a, b + 1)]
            markers.markers.append(line)

            dots = Marker()
            dots.ns = "points"
            dots.id = mid
            mid += 1
            dots.type = Marker.SPHERE_LIST
            dots.action = Marker.ADD
            dots.scale.x = dots.scale.y = dots.scale.z = self.point_size
            dots.color = rgba(colour, 1.0)
            dots.pose.orientation.w = 1.0
            dots.points = list(line.points)
            markers.markers.append(dots)

            # Heading ticks, as a LINE_LIST -- cheaper than one arrow each and
            # readable at this scale.
            ticks = Marker()
            ticks.ns = "heading"
            ticks.id = mid
            mid += 1
            ticks.type = Marker.LINE_LIST
            ticks.action = Marker.ADD
            ticks.scale.x = self.line_width * 0.6
            ticks.color = rgba(colour, 0.55)
            ticks.pose.orientation.w = 1.0
            for k in range(a, b + 1):
                x, y, yaw = self.xs[k], self.ys[k], self.yaws[k]
                ticks.points.append(Point(x=x, y=y, z=0.0))
                ticks.points.append(Point(
                    x=x + self.arrow_length * math.cos(yaw),
                    y=y + self.arrow_length * math.sin(yaw),
                    z=0.0))
            markers.markers.append(ticks)

        # Cusp spheres and labels
        for i in range(len(self.segments) - 1):
            j = self.segments[i][1]
            sphere = Marker()
            sphere.ns = "cusp"
            sphere.id = mid
            mid += 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = self.xs[j]
            sphere.pose.position.y = self.ys[j]
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = self.point_size * 3.5
            sphere.color = rgba(CUSP_RGB, 0.9)
            markers.markers.append(sphere)

            markers.markers.append(self._label(
                mid, self.xs[j], self.ys[j] + 0.18, f"cusp (row {j})", CUSP_RGB))
            mid += 1

        markers.markers.append(self._label(
            mid, self.xs[0], self.ys[0] + 0.18, "START", FORWARD_RGB))
        mid += 1
        markers.markers.append(self._label(
            mid, self.xs[-1], self.ys[-1] + 0.18, "END",
            FORWARD_RGB if self.dirs[-1] > 0 else REVERSE_RGB))
        mid += 1

        self.msg_markers = markers

    def _label(self, mid, x, y, text, colour):
        m = Marker()
        m.ns = "labels"
        m.id = mid
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.05
        m.pose.orientation.w = 1.0
        m.scale.z = self.label_size
        m.color = rgba(colour, 1.0)
        m.text = text
        return m

    # --------------------------------------------------------------- publish

    def _publish(self):
        self._stamp_header(self.msg_path)
        self._stamp_header(self.msg_fwd)
        self._stamp_header(self.msg_rev)
        for m in self.msg_markers.markers:
            self._stamp_header(m)

        self.pub_path.publish(self.msg_path)
        self.pub_fwd.publish(self.msg_fwd)
        self.pub_rev.publish(self.msg_rev)
        self.pub_markers.publish(self.msg_markers)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PathViz()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()