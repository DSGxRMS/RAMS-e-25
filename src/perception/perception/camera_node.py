#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


# ----------------------------
# YOLO PATH
# ----------------------------
YOLO_PATH = Path(__file__).parent / "yolov11" / "weights" / "best.onnx"


# ----------------------------
# YOLO helpers
# ----------------------------
def letterbox(im_bgr, new_shape=736, color=(114, 114, 114)):
    h, w = im_bgr.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        im_bgr = cv2.resize(im_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im_bgr = cv2.copyMakeBorder(
        im_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im_bgr, r, (left, top)


def nms_xyxy(boxes, scores, iou_thresh=0.7):
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1e-6) * (y2 - y1 + 1e-6)

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


class YoloOnnxRunner:
    """
    Ultralytics YOLO ONNX export output assumed: [x,y,w,h, class_scores...]
    Handles shapes: (1,N,4+nc) or (1,4+nc,N)
    """
    def __init__(self, onnx_path, imgsz=736, conf=0.65, iou=0.7,
                 class_names=None, use_cuda=True):
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.class_names = class_names or ["yellow", "blue", "orange", "large_orange"]
        self.nc = len(self.class_names)

        providers = ["CPUExecutionProvider"]
        if use_cuda:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)

        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def infer(self, bgr):
        orig_h, orig_w = bgr.shape[:2]

        lb, r, (padx, pady) = letterbox(bgr, self.imgsz)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)

        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # 1x3xHxW

        y = self.sess.run([self.output_name], {self.input_name: x})[0]
        y = np.asarray(y)
        if y.ndim == 3:
            y = y[0]

        # Normalize to (N, 4+nc)
        if y.shape[0] == (4 + self.nc) and y.shape[1] > 10:
            y = y.transpose(1, 0)
        elif y.shape[-1] == (4 + self.nc):
            pass
        else:
            raise RuntimeError(f"Unexpected ONNX output shape: {y.shape}")

        boxes_xywh = y[:, :4]
        cls_scores = y[:, 4:4 + self.nc]

        cls_id = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(cls_scores.shape[0]), cls_id]

        keep = scores >= self.conf
        boxes_xywh = boxes_xywh[keep]
        cls_id = cls_id[keep]
        scores = scores[keep]

        if boxes_xywh.shape[0] == 0:
            return []

        cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # unletterbox to original coords
        boxes[:, [0, 2]] -= padx
        boxes[:, [1, 3]] -= pady
        boxes /= r

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

        # per-class NMS
        dets = []
        for c in range(self.nc):
            idx = np.where(cls_id == c)[0]
            if idx.size == 0:
                continue
            b = boxes[idx]
            s = scores[idx]
            k = nms_xyxy(b, s, self.iou)
            for j in k:
                x1, y1, x2, y2 = b[j]
                dets.append((int(c), float(s[j]), int(x1), int(y1), int(x2), int(y2)))
        return dets


# ----------------------------
# Node: YOLO + Stereo depth -> publish cone points
# ----------------------------
class StereoYoloConePublisher(Node):
    def __init__(self):
        super().__init__("stereo_yolo_cone_publisher")
        self.bridge = CvBridge()

        # Topics
        self.declare_parameter("left_image", "/zed/left/image_rect_color")
        self.declare_parameter("right_image", "/zed/right/image_rect_color")
        self.declare_parameter("left_info", "/zed/left/camera_info")
        self.declare_parameter("right_info", "/zed/right/camera_info")
        self.declare_parameter("output_topic", "/perception/cones_stereo")

        # YOLO params
        self.declare_parameter("imgsz", 736)
        self.declare_parameter("conf", 0.65)
        self.declare_parameter("iou_nms", 0.7)
        self.declare_parameter("use_cuda", True)
        self.declare_parameter("class_names", ["yellow", "blue", "orange", "large_orange"])

        # Stereo params
        self.declare_parameter("num_disparities", 128)  # multiple of 16
        self.declare_parameter("block_size", 7)         # odd
        self.declare_parameter("min_disparity", 0)
        self.declare_parameter("disp_min_valid", 0.5)
        self.declare_parameter("depth_patch", 7)        # median patch size
        self.declare_parameter("max_depth_m", 40.0)

        # Performance
        self.declare_parameter("disparity_downscale", 1)  # 1 or 2 (2 = faster, lower detail)

        # Output frame mode
        # "optical": X right, Y down, Z forward
        # "ros":     x forward, y left, z up
        self.declare_parameter("output_frame_mode", "optical")

        # Internal calibration
        self.P_left = None
        self.P_right = None
        self.fx = self.fy = self.cx = self.cy = None
        self.baseline = None

        # Subscribe camera infos
        self.create_subscription(CameraInfo, self.get_parameter("left_info").value,
                                 self.cb_left_info, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.get_parameter("right_info").value,
                                 self.cb_right_info, qos_profile_sensor_data)

        # YOLO runner
        self.class_names = list(self.get_parameter("class_names").value)
        self.yolo = YoloOnnxRunner(
            onnx_path=YOLO_PATH,
            imgsz=int(self.get_parameter("imgsz").value),
            conf=float(self.get_parameter("conf").value),
            iou=float(self.get_parameter("iou_nms").value),
            class_names=self.class_names,
            use_cuda=bool(self.get_parameter("use_cuda").value),
        )

        # Sync stereo images
        self.sub_l = Subscriber(self, Image, self.get_parameter("left_image").value,
                                qos_profile=qos_profile_sensor_data)
        self.sub_r = Subscriber(self, Image, self.get_parameter("right_image").value,
                                qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer([self.sub_l, self.sub_r], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.cb_stereo)

        # Publisher: PointCloud2 with x,y,z,class_id
        self.pub = self.create_publisher(PointCloud2, self.get_parameter("output_topic").value,
                                         qos_profile_sensor_data)

        self.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="class_id", offset=12, datatype=PointField.UINT32, count=1),
        ]

        # Fixed mapping: YOLO -> class_id for your plotter
        # 0 yellow, 1 blue, 2 orange, 3 large_orange
        self.YOLO_TO_CLASSID = {0: 0, 1: 1, 2: 2, 3: 3}

    def cb_left_info(self, msg: CameraInfo):
        self.P_left = np.array(msg.p, dtype=np.float64).reshape(3, 4)
        self._try_init_calib()

    def cb_right_info(self, msg: CameraInfo):
        self.P_right = np.array(msg.p, dtype=np.float64).reshape(3, 4)
        self._try_init_calib()

    def _try_init_calib(self):
        if self.P_left is None or self.P_right is None:
            return

        self.fx = float(self.P_left[0, 0])
        self.fy = float(self.P_left[1, 1])
        self.cx = float(self.P_left[0, 2])
        self.cy = float(self.P_left[1, 2])

        pl03 = float(self.P_left[0, 3])
        pr03 = float(self.P_right[0, 3])
        diff = abs(pr03 - pl03)

        if diff > 1e-6:
            self.baseline = diff / self.fx
        else:
            self.baseline = max(abs(pr03), abs(pl03)) / self.fx

    def _build_sgbm(self, num_disp, block, min_disp):
        return cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block,
            P1=8 * 1 * block * block,
            P2=32 * 1 * block * block,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def _depth_at(self, disp, u, v, patch, fx_eff, baseline):
        H, W = disp.shape[:2]
        r = patch // 2
        u0, u1 = max(0, u - r), min(W, u + r + 1)
        v0, v1 = max(0, v - r), min(H, v + r + 1)
        d = disp[v0:v1, u0:u1].reshape(-1)

        dmin = float(self.get_parameter("disp_min_valid").value)
        d = d[d > dmin]
        if d.size == 0:
            return None

        d_med = float(np.median(d))
        Z = (fx_eff * baseline) / d_med

        maxZ = float(self.get_parameter("max_depth_m").value)
        if not np.isfinite(Z) or Z <= 0.0 or Z > maxZ:
            return None
        return Z

    def _make_cloud(self, header_in, frame_id, pts):
        point_step = 16
        data = bytearray(point_step * len(pts))
        for i, (x, y, z, cls_id) in enumerate(pts):
            base = i * point_step
            data[base + 0: base + 4] = np.float32(x).tobytes()
            data[base + 4: base + 8] = np.float32(y).tobytes()
            data[base + 8: base + 12] = np.float32(z).tobytes()
            data[base + 12: base + 16] = np.uint32(cls_id).tobytes()

        msg = PointCloud2()
        msg.header = header_in
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = len(pts)
        msg.fields = self.fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = point_step * len(pts)
        msg.is_dense = True
        msg.data = bytes(data)
        return msg

    def cb_stereo(self, left_msg: Image, right_msg: Image):
        if self.fx is None or self.baseline is None:
            return

        left_bgr = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="bgr8")
        right_bgr = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding="bgr8")
        H0, W0 = left_bgr.shape[:2]

        # disparity downscale
        ds = int(self.get_parameter("disparity_downscale").value)
        ds = 1 if ds not in (1, 2) else ds

        if ds == 2:
            left_small = cv2.resize(left_bgr, (W0 // 2, H0 // 2), interpolation=cv2.INTER_AREA)
            right_small = cv2.resize(right_bgr, (W0 // 2, H0 // 2), interpolation=cv2.INTER_AREA)
            left_g = cv2.cvtColor(left_small, cv2.COLOR_BGR2GRAY)
            right_g = cv2.cvtColor(right_small, cv2.COLOR_BGR2GRAY)
            fx_eff = self.fx / 2.0
            baseline = self.baseline
        else:
            left_g = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
            right_g = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
            fx_eff = self.fx
            baseline = self.baseline

        # SGBM setup
        num_disp = int(self.get_parameter("num_disparities").value)
        num_disp = max(16, (num_disp // 16) * 16)
        block = int(self.get_parameter("block_size").value)
        if block % 2 == 0:
            block += 1
        block = max(3, block)
        min_disp = int(self.get_parameter("min_disparity").value)
        sgbm = self._build_sgbm(num_disp, block, min_disp)

        # disparity once
        disp = sgbm.compute(left_g, right_g).astype(np.float32) / 16.0

        # YOLO on left (full res)
        dets = self.yolo.infer(left_bgr)

        patch = int(self.get_parameter("depth_patch").value)
        if patch % 2 == 0:
            patch += 1

        out_mode = str(self.get_parameter("output_frame_mode").value).lower()
        pts = []

        for (cid, score, x1, y1, x2, y2) in dets:
            # bbox clamp
            x1 = int(np.clip(x1, 0, W0 - 1))
            x2 = int(np.clip(x2, 0, W0 - 1))
            y1 = int(np.clip(y1, 0, H0 - 1))
            y2 = int(np.clip(y2, 0, H0 - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            # bottom-center pixel
            u0 = int(0.5 * (x1 + x2))
            v0 = int(y2 - 2)
            u0 = int(np.clip(u0, 0, W0 - 1))
            v0 = int(np.clip(v0, 0, H0 - 1))

            # map to disp res if downscaled
            u = u0 // ds
            v = v0 // ds

            Z = self._depth_at(disp, u, v, patch, fx_eff, baseline)
            if Z is None:
                continue

            # back-project in optical frame using original intrinsics
            X = (u0 - self.cx) * Z / self.fx
            Y = (v0 - self.cy) * Z / self.fy

            if out_mode == "ros":
                # optical -> ROS camera coordinates
                x_out = Z
                y_out = -X
                z_out = -Y
                frame_id = "zed_left_camera_frame"
            else:
                x_out, y_out, z_out = X, Y, Z
                frame_id = "zed_left_camera_optical_frame"

            cls_id = self.YOLO_TO_CLASSID.get(int(cid), 4)
            pts.append((float(x_out), float(y_out), float(z_out), int(cls_id)))

        cloud = self._make_cloud(left_msg.header, frame_id, pts)
        self.pub.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = StereoYoloConePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
