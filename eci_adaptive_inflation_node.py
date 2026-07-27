#!/usr/bin/env python3
"""
eci_adaptive_inflation_node.py  (v3 - corridor-width cap + richer instrumentation)
------------------------------------------------------------------------------------
ECI formula UNCHANGED from the previous version:

    ECI = 0.30*speed + 0.25*obstacle_density + 0.20*localization
          + 0.10*stuck_factor - 0.15*free_space

NEW in this version:

1. CORRIDOR-WIDTH SAFETY CAP (Option 1 from design discussion)
   IR is computed exactly as before, then clamped:
       inflation_radius = min(IR_raw, CAP_FRACTION * corridor_width)
   This is a post-processing safety guard only -- it does NOT change the ECI
   formula or its weights. corridor_width is estimated the same way as in
   earlier versions (left+right LiDAR beam proxy), reinstated here solely
   for this cap.

2. RICHER LOGGING for report-quality analysis:
   - corridor_width, ir_before_cap, ir_after_cap, cap_active (bool) logged
     every tick -> lets you quantify exactly how often/how much the cap
     actually intervenes, per environment.
   - Goal outcome tracking via /navigate_to_pose/_action/status
     (GoalStatusArray) -> real SUCCEEDED / ABORTED / CANCELED result per run,
     something the previous version never captured.
   - Backup/recovery event counter via /cmd_vel (linear.x < -0.01 counts as
     a backing-up event) -> a concrete, countable recovery-behavior metric.

Run this ALONGSIDE Nav2 (after it's fully up), before you send a goal and
start bag recording, for your "Proposed Method" experiment runs.

Usage:
    python3 eci_adaptive_inflation_node.py
"""

import math
from collections import deque
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from action_msgs.msg import GoalStatusArray

from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType


# ------------------------------------------------------------------
# Robot / sensor constants
# ------------------------------------------------------------------
LIDAR_MAX_RANGE = 3.5
NEAR_FIELD = 2.0
MAX_LIN_VEL = 0.22
TB3_FOOTPRINT_RADIUS = 0.105

# ------------------------------------------------------------------
# ECI weights - UNCHANGED
# ------------------------------------------------------------------
W_SPEED = 0.30
W_DENSITY = 0.25
W_LOCAL = 0.20
W_STUCK = 0.10
W_FREE_SPACE = 0.15   # subtracted

# ------------------------------------------------------------------
# Inflation Radius
# ------------------------------------------------------------------
R_MIN = 0.15
R_MAX = 0.55

# NEW: corridor-width safety cap (Option 1)
CAP_FRACTION = 0.40          # IR never exceeds this fraction of corridor width
CORRIDOR_NORM_MIN = 0.4      # only used for sanity clamping the raw estimate
CORRIDOR_NORM_MAX = 6.0

LOC_COV_NORM_MAX = 3.0

STUCK_WINDOW_SIZE = 15
STUCK_SPEED_THRESHOLD = 0.02

BACKUP_LINEAR_THRESHOLD = -0.01   # cmd_vel.linear.x below this = backing up

UPDATE_PERIOD_SEC = 2.0

# GoalStatus constants (action_msgs/msg/GoalStatus)
STATUS_SUCCEEDED = 4
STATUS_CANCELED = 5
STATUS_ABORTED = 6
STATUS_NAMES = {0: "UNKNOWN", 1: "ACCEPTED", 2: "EXECUTING", 3: "CANCELING",
                4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}


def normalize(x, lo, hi):
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def classify(eci):
    if eci < 0.35:
        return "Low"
    elif eci < 0.65:
        return "Medium"
    return "High"


class ECIAdaptiveInflationNode(Node):
    def __init__(self):
        super().__init__("eci_adaptive_inflation_node")

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        status_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.latest_speed = 0.0
        self.latest_density = 0.0
        self.latest_avg_range = LIDAR_MAX_RANGE
        self.latest_corridor_width = CORRIDOR_NORM_MAX
        self.latest_loc_cov = 0.0
        self.speed_history = deque(maxlen=STUCK_WINDOW_SIZE)

        self.n_backup_events = 0
        self._was_backing_up = False
        self.last_goal_status = None
        self.goal_status_log = []

        self.create_subscription(LaserScan, "/scan", self.scan_cb, sensor_qos)
        self.create_subscription(Odometry, "/odom", self.odom_cb, sensor_qos)
        self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self.pose_cb, sensor_qos
        )
        self.create_subscription(Twist, "/cmd_vel", self.cmdvel_cb, sensor_qos)
        self.create_subscription(
            GoalStatusArray, "/navigate_to_pose/_action/status", self.goal_status_cb, status_qos
        )

        self.local_client = self.create_client(
            SetParameters, "/local_costmap/local_costmap/set_parameters"
        )
        self.global_client = self.create_client(
            SetParameters, "/global_costmap/global_costmap/set_parameters"
        )

        self.get_logger().info("Waiting for costmap parameter services...")
        self.local_client.wait_for_service(timeout_sec=10.0)
        self.global_client.wait_for_service(timeout_sec=10.0)
        self.get_logger().info("Costmap services found. Starting adaptive inflation loop.")

        self.timer = self.create_timer(UPDATE_PERIOD_SEC, self.update_inflation)
        self.log_rows = []

    # ---------------- sensor callbacks ----------------
    def scan_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float64)
        valid = ranges[np.isfinite(ranges) & (ranges > 0)]
        if len(valid) == 0:
            return

        close_hits = np.sum(valid < NEAR_FIELD)
        self.latest_density = float(close_hits / len(valid))

        capped = np.minimum(valid, LIDAR_MAX_RANGE)
        self.latest_avg_range = float(np.mean(capped))

        # corridor width proxy (left+right beam distance), reinstated ONLY for the cap
        n = len(ranges)
        if n >= 4:
            left = ranges[n // 4]
            right = ranges[3 * n // 4]
            if np.isfinite(left) and np.isfinite(right) and left > 0 and right > 0:
                width = min(left + right, CORRIDOR_NORM_MAX)
                self.latest_corridor_width = max(width, CORRIDOR_NORM_MIN)

    def odom_cb(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        speed = math.sqrt(vx**2 + vy**2)
        self.latest_speed = speed
        self.speed_history.append(speed)

    def pose_cb(self, msg: PoseWithCovarianceStamped):
        cov = msg.pose.covariance
        self.latest_loc_cov = float(cov[0]) + float(cov[7])

    def cmdvel_cb(self, msg: Twist):
        is_backing_up = msg.linear.x < BACKUP_LINEAR_THRESHOLD
        if is_backing_up and not self._was_backing_up:
            self.n_backup_events += 1
            self.get_logger().info(f"Backup event detected (#{self.n_backup_events})")
        self._was_backing_up = is_backing_up

    def goal_status_cb(self, msg: GoalStatusArray):
        for status in msg.status_list:
            name = STATUS_NAMES.get(status.status, f"UNKNOWN({status.status})")
            if name != self.last_goal_status:
                self.last_goal_status = name
                self.goal_status_log.append((self.get_clock().now().to_msg().sec, name))
                self.get_logger().info(f"Goal status -> {name}")

    # ---------------- stuck factor ----------------
    def compute_stuck_factor(self):
        if len(self.speed_history) == 0:
            return 0.0
        stationary_count = sum(1 for s in self.speed_history if s < STUCK_SPEED_THRESHOLD)
        return stationary_count / len(self.speed_history)

    # ---------------- ECI / IR computation ----------------
    def compute_eci_and_ir(self):
        speed_term = normalize(self.latest_speed, 0.0, MAX_LIN_VEL)
        density_term = normalize(self.latest_density, 0.0, 1.0)
        localization_term = normalize(self.latest_loc_cov, 0.0, LOC_COV_NORM_MAX)
        stuck_term = self.compute_stuck_factor()
        free_space_term = normalize(self.latest_avg_range, 0.0, LIDAR_MAX_RANGE)

        eci = (W_SPEED * speed_term + W_DENSITY * density_term + W_LOCAL * localization_term
               + W_STUCK * stuck_term - W_FREE_SPACE * free_space_term)
        eci = max(0.0, min(1.0, eci))

        ir_raw = R_MIN + (R_MAX - R_MIN) * eci

        # NEW: corridor-width safety cap
        ir_cap = CAP_FRACTION * self.latest_corridor_width
        ir_final = min(ir_raw, ir_cap)
        ir_final = max(ir_final, TB3_FOOTPRINT_RADIUS + 0.02)
        cap_active = ir_raw > ir_cap

        return {
            "eci": eci, "ir_raw": ir_raw, "ir_final": ir_final, "cap_active": cap_active,
            "speed_term": speed_term, "density_term": density_term,
            "localization_term": localization_term, "stuck_term": stuck_term,
            "free_space_term": free_space_term, "corridor_width": self.latest_corridor_width,
        }

    # ---------------- push to costmaps ----------------
    def set_inflation_radius(self, client, value):
        req = SetParameters.Request()
        param = Parameter()
        param.name = "inflation_layer.inflation_radius"
        param.value = ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE, double_value=float(value)
        )
        req.parameters = [param]
        return client.call_async(req)

    def update_inflation(self):
        r = self.compute_eci_and_ir()
        eci_class = classify(r["eci"])

        self.set_inflation_radius(self.local_client, r["ir_final"])
        self.set_inflation_radius(self.global_client, r["ir_final"])

        cap_note = " [CAP ACTIVE]" if r["cap_active"] else ""
        self.get_logger().info(
            f"ECI={r['eci']:.3f} ({eci_class}) -> IR_raw={r['ir_raw']:.3f}m "
            f"IR_final={r['ir_final']:.3f}m corridor={r['corridor_width']:.2f}m{cap_note} "
            f"| backups={self.n_backup_events}"
        )

        self.log_rows.append([
            self.get_clock().now().to_msg().sec,
            self.latest_speed, self.latest_density, self.latest_loc_cov, r["corridor_width"],
            r["speed_term"], r["density_term"], r["localization_term"], r["stuck_term"], r["free_space_term"],
            r["eci"], eci_class, r["ir_raw"], r["ir_final"], r["cap_active"], self.n_backup_events,
        ])


def main():
    rclpy.init()
    node = ECIAdaptiveInflationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        import csv
        with open("eci_live_log.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t_sec", "raw_speed", "raw_density", "raw_loc_cov", "corridor_width",
                "speed_term", "density_term", "localization_term", "stuck_term", "free_space_term",
                "eci", "eci_class", "ir_raw", "ir_final", "cap_active", "n_backup_events_cumulative"
            ])
            writer.writerows(node.log_rows)
        node.get_logger().info(f"Saved eci_live_log.csv ({len(node.log_rows)} ticks)")

        with open("goal_status_log.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t_sec", "status"])
            writer.writerows(node.goal_status_log)
        node.get_logger().info(f"Saved goal_status_log.csv, final status: {node.last_goal_status}")
        node.get_logger().info(f"Total backup events this run: {node.n_backup_events}")

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
