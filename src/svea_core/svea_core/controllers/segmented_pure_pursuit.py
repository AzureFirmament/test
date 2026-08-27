#! /usr/bin/env python3
"""
Segmented Pure Pursuit for paths with cusps (direction reversals).

Place in: svea_core/svea_core/controllers/segmented_pure_pursuit.py

Two things make this different from the plain PurePursuitController already in
svea_core:

1. The path is split into monotonic-direction segments at every cusp. A single
   controller instance never tracks across a cusp, because the look-ahead point
   would land on the segment travelling the other way and the geometry becomes
   meaningless.

2. Forward and reverse use different reference points.

   Forward: the rear axle is the "trailing" point and the standard pure pursuit
   law is stable.

       delta = atan2(2 * L * sin(alpha), Ld)

   Reverse: substitute s = -v > 0 and h = yaw + pi. Then

       x_dot = s*cos(h),  y_dot = s*sin(h),  h_dot = -s*tan(delta)/L

   which is a standard forward bicycle model with steering (-delta). So run
   ordinary pure pursuit on the same rear-axle path using heading h, then
   negate:

       alpha = wrap(bearing_to_target - yaw - pi)
       delta = -atan2(2 * L * sin(alpha), Ld)

   Do NOT use the front axle as the reference point here. Pointing the front
   axle at the look-ahead target does make the front axle converge, but the
   body heading is an unbounded zero dynamic under that law -- in simulation
   it walks off monotonically (75 deg -> 153 deg over one 0.9 m segment).

   Footnote: sin(alpha) = -sin(alpha_forward), so the expression above reduces
   to exactly the forward formula. The `alpha = pi - alpha` line in the stock
   svea_core PurePursuitController is therefore a no-op (sin(pi-a) == sin(a)),
   which happens to land on the right answer for the wrong reason. What
   actually matters for reverse is that the look-ahead target is picked along
   the direction of travel.
"""

import csv
import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "wrap_to_pi",
    "PathSegment",
    "load_segments_from_csv",
    "SegmentPurePursuit",
]


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class PathSegment:
    """One constant-direction piece of the path.

    Attributes:
        xs, ys, yaws: rear-axle reference path, map frame.
        direction: +1 forward, -1 reverse.
        wheelbase: needed to build the front-axle offset path for reverse.
    """

    xs: np.ndarray
    ys: np.ndarray
    yaws: np.ndarray
    direction: int
    wheelbase: float = 0.324

    s: np.ndarray = field(init=False)
    curvature: np.ndarray = field(init=False)
    txs: np.ndarray = field(init=False)  # tracking path (rear axle)
    tys: np.ndarray = field(init=False)

    def __post_init__(self):
        self.xs = np.asarray(self.xs, dtype=float)
        self.ys = np.asarray(self.ys, dtype=float)
        self.yaws = np.asarray(self.yaws, dtype=float)

        ds = np.hypot(np.diff(self.xs), np.diff(self.ys))
        self.s = np.concatenate([[0.0], np.cumsum(ds)])

        # Signed curvature from heading change per unit arclength.
        k = np.zeros_like(self.xs)
        for i in range(len(self.xs) - 1):
            if ds[i] > 1e-6:
                k[i] = wrap_to_pi(self.yaws[i + 1] - self.yaws[i]) / ds[i]
        if len(k) > 1:
            k[-1] = k[-2]
        self.curvature = k

        # The CSV describes the rear axle, and both the forward and the reverse
        # law reference the rear axle. Kept as a separate alias so downstream
        # code reads clearly.
        self.txs, self.tys = self.xs.copy(), self.ys.copy()

    def __len__(self) -> int:
        return len(self.xs)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def is_reverse(self) -> bool:
        return self.direction < 0

    @property
    def travel_unit(self) -> tuple:
        """Unit vector of travel at the end of the segment (tracking path)."""
        if len(self) < 2:
            return (math.cos(self.yaws[-1]), math.sin(self.yaws[-1]))
        ex = self.txs[-1] - self.txs[-2]
        ey = self.tys[-1] - self.tys[-2]
        n = math.hypot(ex, ey)
        if n < 1e-9:
            return (math.cos(self.yaws[-1]), math.sin(self.yaws[-1]))
        return (ex / n, ey / n)

    def max_required_steering(self) -> float:
        """Largest steering angle [rad] this segment demands. Feasibility check."""
        return float(np.max(np.arctan(self.wheelbase * np.abs(self.curvature))))


def load_segments_from_csv(csv_path: str,
                           wheelbase: float = 0.324,
                           min_segment_length: float = 0.05) -> list:
    """Read a path CSV and split it into constant-direction segments.

    Expected columns: x, y, yaw, direction.

    The cusp point itself is duplicated: it is both the last point of the
    outgoing segment and the first point of the incoming one, so the next
    segment starts exactly where the vehicle came to rest.
    """
    xs, ys, yaws, dirs = [], [], [], []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            yaws.append(float(row["yaw"]))
            dirs.append(int(round(float(row["direction"]))))

    if len(xs) < 2:
        raise ValueError(f"Path '{csv_path}' has fewer than 2 points")

    segments = []
    start = 0
    for i in range(1, len(dirs) + 1):
        if i == len(dirs) or dirs[i] != dirs[start]:
            sx = xs[start:i]
            sy = ys[start:i]
            sw = yaws[start:i]

            # Prepend the cusp point from the previous segment.
            if segments:
                sx = [float(segments[-1].xs[-1])] + sx
                sy = [float(segments[-1].ys[-1])] + sy
                sw = [sw[0]] + sw

            if len(sx) >= 2:
                seg = PathSegment(np.array(sx), np.array(sy), np.array(sw),
                                  dirs[start], wheelbase)
                if seg.length >= min_segment_length:
                    segments.append(seg)
            start = i

    if not segments:
        raise ValueError(f"No usable segments in '{csv_path}'")
    return segments


class SegmentPurePursuit:
    """Pure pursuit for a single PathSegment.

    Call set_segment() then compute() at a fixed rate. compute() returns
    (steering [rad], speed [m/s, signed], info dict).
    """

    SEARCH_WINDOW = 40  # points; keeps the nearest-point search from jumping
                        # backwards on paths that nearly self-intersect

    def __init__(self,
                 wheelbase: float = 0.324,
                 lookahead_base: float = 0.35,
                 lookahead_gain: float = 0.30,
                 lookahead_min: float = 0.25,
                 lookahead_max: float = 0.70,
                 max_steering: float = 0.45,
                 max_steering_rate: float = 3.0,
                 target_speed: float = 0.35,
                 min_speed: float = 0.15,
                 max_lateral_accel: float = 1.2,
                 decel_distance: float = 0.45,
                 goal_tolerance: float = 0.08):
        self.L = wheelbase
        self.Lfc = lookahead_base
        self.k = lookahead_gain
        self.Ld_min = lookahead_min
        self.Ld_max = lookahead_max
        self.max_steering = max_steering
        self.max_steering_rate = max_steering_rate
        self.target_speed = target_speed
        self.min_speed = min_speed
        self.a_lat_max = max_lateral_accel
        self.decel_distance = decel_distance
        self.goal_tolerance = goal_tolerance

        self.seg = None
        self._idx = 0
        self._last_steering = 0.0

    def set_segment(self, seg: PathSegment) -> None:
        self.seg = seg
        self._idx = 0
        self._last_steering = 0.0

    def initial_steering(self) -> float:
        """Steering the segment asks for at its first point.

        Useful for pre-steering while stopped at a cusp.
        """
        if self.seg is None:
            return 0.0
        delta = math.atan(self.L * float(self.seg.curvature[0]))
        if self.seg.is_reverse:
            delta = -delta
        return float(np.clip(delta, -self.max_steering, self.max_steering))

    def compute(self, x: float, y: float, yaw: float, v: float, dt: float):
        """One control step.

        Args:
            x, y, yaw: rear-axle pose in the map frame.
            v: measured longitudinal speed [m/s] (magnitude is what matters).
            dt: control period [s], used for the steering rate limit.
        """
        seg = self.seg
        if seg is None:
            return 0.0, 0.0, {"finished": True, "reason": "no segment"}

        # --- reference point ---------------------------------------------
        # Rear axle in both directions (see module docstring).
        rx, ry = x, y

        # --- look-ahead distance -----------------------------------------
        Ld = float(np.clip(self.k * abs(v) + self.Lfc, self.Ld_min, self.Ld_max))

        # --- nearest point, forward-windowed ------------------------------
        n = len(seg)
        lo = self._idx
        hi = min(n, self._idx + self.SEARCH_WINDOW)
        d = np.hypot(seg.txs[lo:hi] - rx, seg.tys[lo:hi] - ry)
        self._idx = lo + int(np.argmin(d))
        cross_track = float(d.min())

        # Tighten the look-ahead as the segment end approaches. A segment that
        # ends mid-curve (which is exactly what happens at a cusp) will
        # otherwise be cut short and the vehicle arrives at the cusp with a
        # large heading error, which then poisons the next segment.
        s0 = seg.s[self._idx]
        arc_left = seg.length - s0
        Ld = float(np.clip(Ld, self.Ld_min, max(self.Ld_min, arc_left)))

        # --- look-ahead target --------------------------------------------
        ti = self._idx
        while ti + 1 < n and (seg.s[ti] - s0) < Ld:
            ti += 1

        tx, ty = float(seg.txs[ti]), float(seg.tys[ti])
        overshoot = Ld - (seg.s[ti] - s0)
        if overshoot > 1e-3:
            # Past the last point: extend a virtual target so the geometry
            # stays valid instead of collapsing onto the endpoint. Extend
            # along the terminal *arc*, not a straight line -- a straight
            # extension flattens out the final curve and costs heading
            # accuracy right where it matters most (the cusp).
            ux, uy = seg.travel_unit
            h = math.atan2(uy, ux)
            k_end = float(seg.curvature[-1])
            if abs(k_end) > 1e-3:
                R = 1.0 / k_end
                h2 = h + k_end * overshoot
                tx += R * (math.sin(h2) - math.sin(h))
                ty -= R * (math.cos(h2) - math.cos(h))
            else:
                tx += ux * overshoot
                ty += uy * overshoot

        # --- steering ------------------------------------------------------
        bearing = math.atan2(ty - ry, tx - rx)
        if seg.is_reverse:
            alpha = wrap_to_pi(bearing - yaw - math.pi)
            delta = -math.atan2(2.0 * self.L * math.sin(alpha), Ld)
        else:
            alpha = wrap_to_pi(bearing - yaw)
            delta = math.atan2(2.0 * self.L * math.sin(alpha), Ld)

        delta = float(np.clip(delta, -self.max_steering, self.max_steering))

        # Steering rate limit -- keeps the servo from slamming and keeps the
        # command differentiable enough for the EKF to stay happy.
        if self.max_steering_rate > 0.0 and dt > 0.0:
            max_step = self.max_steering_rate * dt
            delta = float(np.clip(delta,
                                  self._last_steering - max_step,
                                  self._last_steering + max_step))
        self._last_steering = delta

        # --- progress / termination ---------------------------------------
        # ux, uy = seg.travel_unit
        # ex, ey = float(seg.txs[-1]) - rx, float(seg.tys[-1]) - ry
        # remaining = ex * ux + ey * uy          # >0 while the end is ahead
        # dist_to_end = math.hypot(ex, ey)

        # finished = (remaining <= 0.0) or (dist_to_end <= self.goal_tolerance)
        ux, uy = seg.travel_unit # 仅 decel ramp 用，保留
        ex, ey = float(seg.txs[-1]) - rx, float(seg.tys[-1]) - ry
        dist_to_end = math.hypot(ex, ey)

        # 用弧长剩余代替投影 -- 对 U 形、绕回、近闭合段都正确
        remaining = seg.length - seg.s[self._idx]
        finished = (remaining <= 0.0) or (dist_to_end <= self.goal_tolerance)

        # --- speed ----------------------------------------------------------
        speed = self.target_speed

        kappa = abs(float(seg.curvature[self._idx]))
        if kappa > 1e-3:
            speed = min(speed, math.sqrt(self.a_lat_max / kappa))

        if self.decel_distance > 1e-6:
            ramp = max(0.0, remaining) / self.decel_distance
            speed = min(speed, max(self.min_speed, self.target_speed * ramp))

        speed = max(speed, self.min_speed)
        signed_speed = 0.0 if finished else speed * seg.direction

        info = {
            "finished": finished,
            "index": self._idx,
            "target_index": ti,
            "target": (tx, ty),
            "lookahead": Ld,
            "cross_track": cross_track,
            "remaining": remaining,
            "dist_to_end": dist_to_end,
            "curvature": float(seg.curvature[self._idx]),
        }
        return delta, signed_speed, info
