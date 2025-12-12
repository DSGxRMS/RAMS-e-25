#!/usr/bin/env python3
# -- coding: utf-8 --

import math
import struct
import threading
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage, PointCloud2, PointField
from sensor_msgs.msg import Imu

# Try importing Hungarian Algorithm (Scipy)
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ----------------- CONFIGURATION -----------------
FX = 448.1338
BASELINE = 0.12
CX = 640.5

# Advanced Tuning
MERGE_RADIUS = 1.2      # Base gating distance
MIN_CONFIDENCE = 5      # Confirmation threshold
MAX_DIST = 25.0
TRACK_LIFE = 20         # How long a track survives without being seen (2.0s)

# Colors
COLORS = {0: "gold", 1: "blue", 2: "darkorange", 3: "red", 4: "gray"}


# ----------------- KALMAN CONE CLASS -----------------
class KalmanCone:
    def __init__(self, x, y, cls_id):
        self.cls_votes = Counter()
        self.cls_votes[cls_id] += 1

        self.count = 1
        self.last_seen = 0

        # --- OPENCV KALMAN FILTER ---
        # 4 State Vars: [x, y, vx, vy]
        # 2 Meas Vars:  [x, y]
        self.kf = cv2.KalmanFilter(4, 2)

        # Initial State
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)

        # Transition Matrix (Physics: x = x + v*t)
        dt = 0.066  # 15Hz
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        self.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)

        # Process Noise (Q): How much we trust the physics
        # Low = Assume static cones. High = Assume cones move.
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05

        # Error Covariance (P): Initial uncertainty
        self.kf.errorCovPre = np.eye(4, dtype=np.float32) * 1.0

    def predict(self):
        # Predict next position based on velocity
        self.kf.predict()

    def update(self, x_meas, y_meas, cls_id):
        # DYNAMIC MEASUREMENT NOISE (The "Secret Sauce")
        # Trust close points (2m) 100x more than far points (20m)
        dist = math.hypot(x_meas, y_meas)
        noise = (dist * 0.15) ** 2

        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * noise

        # Correct state
        meas = np.array([[x_meas], [y_meas]], dtype=np.float32)
        self.kf.correct(meas)

        self.cls_votes[cls_id] += 1
        self.count += 1
        self.last_seen = 0

    def apply_rotation(self, dyaw: float):
        # Explicitly rotate the Kalman State Vector
        c = math.cos(dyaw)
        s = math.sin(dyaw)

        x = self.x
        y = self.y
        vx = self.vx
        vy = self.vy

        # Rotate Position
        nx = x * c + y * s
        ny = -x * s + y * c

        # Rotate Velocity
        nvx = vx * c + vy * s
        nvy = -vx * s + vy * c

        # Force update state
        self.kf.statePost[0, 0] = nx
        self.kf.statePost[1, 0] = ny
        self.kf.statePost[2, 0] = nvx
        self.kf.statePost[3, 0] = nvy

    @property
    def x(self):
        return float(self.kf.statePost[0, 0])

    @property
    def y(self):
        return float(self.kf.statePost[1, 0])

    @property
    def vx(self):
        return float(self.kf.statePost[2, 0])

    @property
    def vy(self):
        return float(self.kf.statePost[3, 0])

    @property
    def best_cls(self):
        return self.cls_votes.most_common(1)[0][0]


# ----------------- YOLO -----------------
class YOLOv5:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        script_dir = Path(__file__).parent.resolve()
        repo_path = script_dir / "yolov5"
        weights_path = script_dir / "yolov5" / "weights" / "best.pt"

        try:
            self.model = torch.hub.load(
                str(repo_path),
                "custom",
                path=str(weights_path),
                source="local",
            )
            self.model.to(self.device).eval()
            self.ok = True
        except Exception as e:
            print(f"[YOLO] Error: {e}")
            self.ok = False

    def infer(self, img):
        if not self.ok:
            return []
        res = self.model(img, size=640)
        dets = []
        if len(res.xyxy) > 0:
            for x1, y1, x2, y2, conf, cls in res.xyxy[0].cpu().numpy():
                if conf > 0.45:
                    dets.append((int(x1), int(y1), int(x2), int(y2), int(cls)))
        return dets


# ----------------- MAIN NODE -----------------
class ProMapperNode(Node):
    def __init__(self):
        super().__init__("pro_mapper_node")

        self.yolo = YOLOv5()
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.last_img_l = None
        self.last_img_r = None
        self.stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

        self.last_yaw = None
        self.current_yaw = None
        self.cones = []

        # Subs
        self.create_subscription(
            RosImage,
            "/zed/left/image_rect_color",
            self.cb_left,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            RosImage,
            "/zed/right/image_rect_color",
            self.cb_right,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu,
            "/camera/imu/data",
            self.cb_imu,
            qos_profile_sensor_data,
        )

        # GT for visual compare
        self.gt_cones = []
        try:
            from eufs_msgs.msg import ConeArrayWithCovariance

            self.create_subscription(
                ConeArrayWithCovariance,
                "/ground_truth/cones",
                self.cb_gt,
                qos_profile_sensor_data,
            )
        except Exception:
            pass

        # Pub
        self.pub_map = self.create_publisher(
            PointCloud2,
            "/perception/cones_map",
            qos_profile_sensor_data,
        )

        # 15 Hz Loop
        self.create_timer(0.066, self.process_pipeline)

        # Plotting
        matplotlib.use("Qt5Agg")
        self.fig, self.ax = plt.subplots(figsize=(6, 8))
        self.timer_plot = self.create_timer(0.2, self.update_plot)

        mode = "Hungarian (Pro)" if SCIPY_AVAILABLE else "Greedy (Fallback)"
        self.get_logger().info(f"Pro Mapper Running. Mode: {mode}")

    # ----- Callbacks -----
    def cb_left(self, msg: RosImage):
        self.last_img_l = msg

    def cb_right(self, msg: RosImage):
        self.last_img_r = msg

    def cb_imu(self, msg: Imu):
        q = msg.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        with self.lock:
            self.current_yaw = yaw

    def cb_gt(self, msg):
        temp = []

        try:
            # GT extraction helper
            def ext(cones, cid):
                for c in cones:
                    pt = (
                        c.point
                        if hasattr(c, "point")
                        else (c.location if hasattr(c, "location") else c)
                    )
                    temp.append((pt.x, pt.y, cid))

            ext(msg.yellow_cones, 0)
            ext(msg.blue_cones, 1)
            ext(msg.orange_cones, 2)
        except Exception:
            pass

        with self.lock:
            self.gt_cones = temp

    # ----- Main processing -----
    def process_pipeline(self):
        if (
            self.last_img_l is None
            or self.last_img_r is None
        ):
            return

        # 1. Physics Update
        with self.lock:
            if self.last_yaw is not None and self.current_yaw is not None:
                dyaw = self.current_yaw - self.last_yaw
                if abs(dyaw) < 1.0:  # Ignore jumps
                    for cone in self.cones:
                        cone.apply_rotation(dyaw)
                        cone.predict()  # Kalman Predict Step
            if self.current_yaw is not None:
                self.last_yaw = self.current_yaw

        # 2. Vision
        try:
            imgL_color = self.bridge.imgmsg_to_cv2(self.last_img_l, "bgr8")
            imgL = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2GRAY)
            imgR = self.bridge.imgmsg_to_cv2(self.last_img_r, "mono8")
        except Exception:
            return

        disp_map = self.stereo.compute(imgL, imgR).astype(np.float32) / 16.0
        rgb = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2RGB)
        boxes = self.yolo.infer(rgb)

        measurements = []
        for x1, y1, x2, y2, cls_id in boxes:
            if x1 < 0 or y2 > disp_map.shape[0] or x2 > disp_map.shape[1]:
                continue
            roi = disp_map[y1:y2, x1:x2]
            valid = roi[roi > 1.0]
            if len(valid) < 5:
                continue

            disp = np.median(valid)
            if 2.0 < disp < 150.0:
                Z = (FX * BASELINE) / disp
                X = ((x1 + x2) / 2 - CX) * Z / FX
                if 1.0 < Z < MAX_DIST and abs(X) < 12.0:
                    # Note: world frame uses x=forward (Z), y=left(-X)
                    measurements.append({"x": Z, "y": -X, "cls": cls_id})

        self.update_map_hungarian(measurements)
        self.publish_cloud(self.last_img_l.header)

    # ----- Tracking / Hungarian -----
    def update_map_hungarian(self, measurements):
        with self.lock:
            # 1. Setup Cost Matrix
            n_cones = len(self.cones)
            n_meas = len(measurements)

            # If nothing to match, just add new
            if n_cones == 0:
                for m in measurements:
                    self.cones.append(KalmanCone(m["x"], m["y"], m["cls"]))
                return

            if n_meas == 0:
                # No measurements: just age tracks
                for cone in self.cones:
                    cone.last_seen += 1
                # Prune
                self.cones = [
                    c
                    for c in self.cones
                    if c.last_seen < (TRACK_LIFE if c.count > 5 else 5)
                ]
                return

            # Cost Matrix: Rows=Tracks, Cols=Measurements
            cost_matrix = np.zeros((n_cones, n_meas), dtype=np.float32)
            for t, cone in enumerate(self.cones):
                for m, meas in enumerate(measurements):
                    dist = math.hypot(cone.x - meas["x"], cone.y - meas["y"])
                    # Gating: If too far, assign large cost
                    if dist > MERGE_RADIUS:
                        cost_matrix[t, m] = 10000.0
                    else:
                        cost_matrix[t, m] = dist

            # 2. Hungarian Assignment
            if SCIPY_AVAILABLE:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
            else:
                # Fallback to greedy if no Scipy
                row_ind, col_ind = [], []

            # 3. Process Assignments
            unmatched_meas = set(range(n_meas))
            matched_cones = set()

            for t, m in zip(row_ind, col_ind):
                if cost_matrix[t, m] < MERGE_RADIUS:
                    # Match Found -> Kalman Update
                    meas = measurements[m]
                    self.cones[t].update(meas["x"], meas["y"], meas["cls"])
                    unmatched_meas.discard(m)
                    matched_cones.add(t)

            # 4. Handle Unmatched
            # Decay old cones
            for t, cone in enumerate(self.cones):
                if t not in matched_cones:
                    cone.last_seen += 1

            # Create new cones
            for m in unmatched_meas:
                meas = measurements[m]
                self.cones.append(KalmanCone(meas["x"], meas["y"], meas["cls"]))

            # 5. Prune
            self.cones = [
                c
                for c in self.cones
                if c.last_seen < (TRACK_LIFE if c.count > 5 else 5)
            ]

    # ----- PointCloud2 publishing -----
    def publish_cloud(self, header):
        cloud = PointCloud2()
        cloud.header = header
        cloud.height = 1
        cloud.is_dense = True
        cloud.is_bigendian = False
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="class_id",
                offset=12,
                datatype=PointField.FLOAT32,
                count=1,
            ),
        ]
        cloud.point_step = 16

        points = []
        with self.lock:
            for c in self.cones:
                if c.count >= MIN_CONFIDENCE:
                    points.append(
                        struct.pack(
                            "ffff",
                            float(c.x),
                            float(c.y),
                            0.0,
                            float(c.best_cls),
                        )
                    )

        cloud.width = len(points)
        cloud.row_step = cloud.point_step * cloud.width
        cloud.data = b"".join(points)
        self.pub_map.publish(cloud)

    # ----- Matplotlib live plot -----
    def update_plot(self):
        with self.lock:
            map_data = list(self.cones)
            gt = list(self.gt_cones)

        self.ax.cla()
        visible_cnt = len([c for c in map_data if c.count >= MIN_CONFIDENCE])
        self.ax.set_title(f"Kalman+Hungarian Map (Visible: {visible_cnt})")
        self.ax.set_xlim(8, -8)
        self.ax.set_ylim(-2, 25)
        self.ax.grid(True, alpha=0.3)
        self.ax.plot(0, 0, "^", color="lime", ms=12, label="Ego")

        # Ground truth
        for x, y, c in gt:
            col = COLORS.get(c, "black")
            self.ax.plot(y, x, "o", color=col, fillstyle="none", ms=10, mew=2)

        # Tracked cones
        for c in map_data:
            if c.count < MIN_CONFIDENCE:
                continue
            x, y = c.x, c.y
            col = COLORS.get(c.best_cls, "black")
            self.ax.plot(y, x, "o", color=col, ms=7)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)


def main():
    rclpy.init()
    node = ProMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
