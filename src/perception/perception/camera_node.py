#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import Image as RosImage, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2

import cv2
import numpy as np
from pathlib import Path
import torch

# ----------------- CONFIGURATION -----------------
FX = 448.1338
BASELINE = 0.12
CX = 640.5

MAX_DIST = 25.0  # Ignore cones further than 25m (too noisy)


# ----------------- YOLO -----------------
class YOLOv5:
    def __init__(self, repo, weights):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = torch.hub.load(str(repo), "custom", path=str(weights), source="local")
            self.model.to(self.device).eval()
            self.ok = True
        except Exception as e:
            print(f"[YOLOv5] Failed to load model: {e}")
            self.ok = False

    def infer(self, img):
        """
        img: RGB numpy image (H, W, 3), same as your original.
        Returns list of (x1, y1, x2, y2, cls_id).
        """
        if not self.ok:
            return []
        res = self.model(img, size=640)
        dets = []
        if len(res.xyxy) > 0:
            for x1, y1, x2, y2, conf, cls in res.xyxy[0].cpu().numpy():
                if conf > 0.50:
                    dets.append((int(x1), int(y1), int(x2), int(y2), int(cls)))
        return dets


# ----------------- MAIN NODE -----------------
class CameraConesNode(Node):
    def __init__(self):
        super().__init__("camera_cones_node")

        base = Path(__file__).parent.resolve()
        self.yolo = YOLOv5(base / "yolov5", base / "yolov5/weights/best.pt")
        self.bridge = CvBridge()

        self.last_img_l = None
        self.last_img_r = None

        # Same StereoBM as your original code
        self.stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

        # Subscriptions (same topics as your mapper)
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

        # Publisher: 2D cones in car frame + class_id
        self.pub_cones = self.create_publisher(
            PointCloud2,
            "/perception/cones_colour",
            qos_profile_sensor_data,
        )

        # Timer for processing
        self.create_timer(0.1, self.process_pipeline)

        # self.get_logger().info("[CameraConesNode] Running (same vision logic, no odom, no map).")

    # ------------- Callbacks -------------
    def cb_left(self, msg: RosImage):
        self.last_img_l = msg

    def cb_right(self, msg: RosImage):
        self.last_img_r = msg

    # ------------- Main pipeline -------------
    def process_pipeline(self):
        if self.last_img_l is None or self.last_img_r is None:
            return

        # 1. Images (1:1 with your original)
        try:
            imgL_color = self.bridge.imgmsg_to_cv2(self.last_img_l, "bgr8")
            imgL = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2GRAY)
            imgR = self.bridge.imgmsg_to_cv2(self.last_img_r, "mono8")
        except Exception as e:
            self.get_logger().warn(f"[CameraConesNode] CvBridge error: {e}")
            return

        if imgL.shape != imgR.shape:
            self.get_logger().warn("[CameraConesNode] Left/right image size mismatch; skipping frame.")
            return

        # 2. Disparity (exactly your code)
        disp_map = self.stereo.compute(imgL, imgR).astype(np.float32) / 16.0

        # 3. YOLO on left RGB (exactly your code)
        rgb = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2RGB)
        boxes = self.yolo.infer(rgb)

        cones = []  # per-frame (x_car, y_car, cls_id)

        for x1, y1, x2, y2, cls_id in boxes:
            # Same bounds check as yours
            if x1 < 0 or y2 > disp_map.shape[0] or x2 > disp_map.shape[1]:
                continue

            roi = disp_map[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            valid = roi[roi > 1.0]
            if len(valid) < 10:
                continue

            disp = float(np.median(valid))
            if 2.0 < disp < 150.0:
                Z = (FX * BASELINE) / disp
                X = ((x1 + x2) / 2.0 - CX) * Z / FX

                x_car = Z
                y_car = -X

                # Same loose range filter as your code
                if 1.0 < x_car < MAX_DIST and abs(y_car) < 12.0:
                    cones.append((x_car, y_car, cls_id))

        # 4. Publish per-frame cones in car frame
        self.publish_cones(self.last_img_l.header, cones)

    # ------------- Publishing -------------
    def publish_cones(self, header, cones):
        """
        cones: list of (x, y, class_id)
        Publish as PointCloud2 with fields:
          - x (float32)
          - y (float32)
          - z (float32, always 0)
          - class_id (uint32)
        """
        fields = [
            PointField(name="x",        offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",        offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",        offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="class_id", offset=12, datatype=PointField.UINT32,  count=1),
        ]

        points = [(float(x), float(y), 0.0, int(cls_id)) for (x, y, cls_id) in cones]

        cloud = pc2.create_cloud(header, fields, points)
        self.pub_cones.publish(cloud)
        # self.get_logger().info(f"[CameraConesNode] Published {len(cones)} cones on /cones_colour.")


def main(args=None):
    rclpy.init(args=args)
    node = CameraConesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
