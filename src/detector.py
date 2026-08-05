# -*- coding: utf-8 -*-
"""detector.py —— 表盘检测 + 关键点推理后端（Top-Down 关键点架构）。

所有后端统一返回“原图像素坐标”的关键点列表，顺序固定为
[min, max, tip, center]（索引 0/1/2/3），下游 geometry / visualizer
无需感知后端差异。

后端：
    - ultralytics : PyTorch 模型（YOLOv8/YOLO11-Pose，训练见 scripts/train_pose.py）
    - onnx        : ONNX 模型（cv2.dnn.readNetFromONNX，工业 PC / 边缘部署首选）
    - tflite      : TFLite 模型（tflite_runtime / tensorflow.lite，RK3588 / Jetson / RPi 首选）
    - mock        : 调试后端，直接使用配置中的关键点走完整读数和可视化链路

设计约定（修订版）：
    - DL 负责表盘检测与关键点；传统图像方法（HoughCircles / findContours /
      Canny / 阈值分割等）仅允许在 refiner.py 中、以 DL 结果为先验做数值精修，
      且失败必须自动回退 DL 结果；
    - OpenCV 在本模块仅用于：图像预处理（letterbox、blobFromImage）、NMS、张量读写。

性能说明：
    - 批量解析全部使用 numpy 向量化操作，避免 Python 层逐锚点循环；
    - 预处理只产生必要的 letterbox 画布，不反复拷贝原始帧；
    - 视频流场景建议开启 ONNX 的 CUDA 后端（cv2.dnn，需 OpenCV 带 CUDA 编译）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

# 关键点顺序约定（与数据集标注顺序完全一致，全链路统一，绝不重排）：
#   kpts[0] = min    刻度起点端（最小值端）
#   kpts[1] = max    刻度终点端（最大值端）
#   kpts[2] = tip    指针尖端
#   kpts[3] = center 表盘中心


def _letterbox(img: np.ndarray, size: int = 640):
    """等比例缩放 + 灰边居中填充到 size x size（与 ultralytics 训练一致）。

    返回 (画布, scale, pad_w, pad_h)：
        scale   原图到画布的缩放比例
        pad_w/pad_h 水平 / 垂直填充偏移
    推理完成后用 scale / pad 把关键点坐标映射回原图。
    """
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas.fill(114)  # YOLO 默认填充色
    pad_w, pad_h = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return canvas, scale, pad_w, pad_h


def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray,
         conf_thres: float = 0.5, iou_thres: float = 0.5) -> List[int]:
    """置信度过滤 + NMS（cv2.dnn.NMSBoxes），返回保留的原始下标列表。"""
    keep = [i for i, s in enumerate(scores) if float(s) >= conf_thres]
    if not keep:
        return []
    # 只取过滤后的候选框，保证 rects 与 scores 长度一致（OpenCV 5 会做断言检查）
    rects = [[float(b[0]), float(b[1]),
              float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes_xyxy[keep]]
    idx = cv2.dnn.NMSBoxes(rects, [float(scores[i]) for i in keep],
                           conf_thres, iou_thres)
    if idx is None or len(idx) == 0:
        return []
    return [keep[int(i)] for i in np.asarray(idx).flatten()]


def _parse_raw_predictions(pred: np.ndarray, num_keypoints: int,
                           conf_thres: float, kpt_conf_thres: float,
                           scale: float, pad_w: int, pad_h: int) -> List[Dict]:
    """解析 ONNX / TFLite 的原始输出张量 (C, N)，返回 Detection 列表（原图像素坐标）。

    输出通道布局（YOLOv8-Pose，未端到端 NMS 的版本）：
        C = 4(bbox cx,cy,w,h) + nc(类别置信度) + 3*K(每个关键点的 x, y, conf)
    """
    n = pred.shape[1]
    channels = pred.shape[0]
    k_channels = 3 * num_keypoints
    nc = channels - 4 - k_channels   # 由通道数动态推断类别数

    boxes_xywh = pred[:4].T                              # (N, 4)
    scores = pred[4]                                     # (N,)
    kpts_raw = pred[5:5 + k_channels].T.reshape(
        n, num_keypoints, 3)                             # (N, K, 3): x, y, conf

    xyxy = np.column_stack([
        boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0,
        boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0,
        boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0,
        boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0,
    ])

    detections: List[Dict] = []
    for i in _nms(xyxy, scores, conf_thres):
        kc = [float(c) for c in kpts_raw[i][:, 2]]
        if min(kc) < kpt_conf_thres:
            continue
        detections.append({
            "bbox": tuple(
                float((v - (pad_w, pad_h, pad_w, pad_h)[j]) / scale)
                for j, v in enumerate(xyxy[i])),
            "keypoints": [
                [float((x - pad_w) / scale), float((y - pad_h) / scale)]
                for x, y in kpts_raw[i][:, :2]],
            "conf": float(scores[i]),
            "kpt_conf": kc,
        })
    return detections


class GaugeDetector(ABC):
    """抽象基类：定义统一推理接口 predict(frame) -> List[Detection]。

    Detection 结构：
        {
            "bbox":      (x1, y1, x2, y2)  原图像素坐标
            "keypoints": [[x, y] * 4]      原图像素坐标，顺序 [min, max, tip, center]
            "conf":      float             检测框置信度
            "kpt_conf":  [float] * 4       各关键点置信度
        }
    """

    def __init__(self, model_cfg: Dict):
        self.conf_thres = float(model_cfg.get("conf_thres", 0.5))
        self.kpt_conf_thres = float(model_cfg.get("kpt_conf_thres", 0.5))
        self.input_size = int(model_cfg.get("input_size", 640))
        self.num_keypoints = int(model_cfg.get("num_keypoints", 4))
        # 由类名推导后端名（UltralyticsDetector -> ultralytics），供外部展示
        self.backend = type(self).__name__.replace("Detector", "").lower()

    @abstractmethod
    def predict(self, frame: np.ndarray) -> List[Dict]:
        """对单帧执行检测 + 关键点推理，返回 Detection 列表（原图像素坐标）。"""
        raise NotImplementedError


class UltralyticsDetector(GaugeDetector):
    """PyTorch 后端：ultralytics.YOLO 原生推理（训练 / 调试阶段常用）。"""

    def __init__(self, model_cfg: Dict):
        super().__init__(model_cfg)
        from ultralytics import YOLO  # 延迟导入，避免其余后端场景产生多余依赖

        path = model_cfg["pt"]["model_path"]
        if not Path(path).exists():
            raise FileNotFoundError(
                f"未找到模型权重: {path}。请先运行 scripts/train_pose.py 训练，"
                "或把 model.backend 切换为 mock / onnx / tflite")
        self.model = YOLO(str(path))

    def predict(self, frame: np.ndarray) -> List[Dict]:
        results = self.model.predict(
            frame, imgsz=self.input_size, conf=self.conf_thres, verbose=False)
        detections: List[Dict] = []
        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                continue
            boxes = res.boxes.xyxy.cpu().numpy()       # (N, 4) 原图坐标
            confs = res.boxes.conf.cpu().numpy()       # (N,)
            kpts = res.keypoints.xy.cpu().numpy()      # (N, K, 2) 原图坐标
            kconf = (res.keypoints.conf.cpu().numpy()
                     if res.keypoints.conf is not None else None)
            for i in range(len(boxes)):
                kc = (kconf[i] if kconf is not None
                      else np.ones(self.num_keypoints))
                if float(np.min(kc)) < self.kpt_conf_thres:
                    continue
                detections.append({
                    "bbox": tuple(float(v) for v in boxes[i]),
                    "keypoints": [[float(x), float(y)] for x, y in kpts[i]],
                    "conf": float(confs[i]),
                    "kpt_conf": [float(c) for c in kc],
                })
        return detections


class ONNXDetector(GaugeDetector):
    """ONNX 后端：cv2.dnn.readNetFromONNX，零额外推理依赖，工业部署首选。

    说明：
        - 预处理与训练保持一致：letterbox -> BGR 转 RGB -> 归一化到 [0,1]；
        - prefer_cuda=true 时优先尝试 CUDA 后端（需 OpenCV 带 CUDA 编译）。
    """

    def __init__(self, model_cfg: Dict):
        super().__init__(model_cfg)
        path = model_cfg["onnx"]["model_path"]
        if not Path(path).exists():
            raise FileNotFoundError(
                f"未找到 ONNX 模型: {path}。请先运行 scripts/train_pose.py --export-onnx")
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.out_names = self.net.getUnconnectedOutLayersNames()

        if model_cfg.get("onnx", {}).get("prefer_cuda", False):
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except Exception as exc:  # OpenCV 未编译 CUDA 支持时静默回退 CPU
                print(f"[detector] CUDA 后端不可用，回退 CPU: {exc}")

    def predict(self, frame: np.ndarray) -> List[Dict]:
        img, scale, pad_w, pad_h = _letterbox(frame, self.input_size)
        blob = cv2.dnn.blobFromImage(
            img, scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            mean=(0, 0, 0), swapRB=True, crop=False)   # BGR -> RGB
        self.net.setInput(blob)
        outs = self.net.forward(self.out_names)
        out = outs[0] if isinstance(outs, (list, tuple)) else outs
        pred = np.squeeze(out)
        if pred.shape[0] > pred.shape[1]:              # (N, C) -> (C, N)
            pred = pred.T
        return _parse_raw_predictions(
            pred, self.num_keypoints, self.conf_thres,
            self.kpt_conf_thres, scale, pad_w, pad_h)


class TFLiteDetector(GaugeDetector):
    """TFLite 后端：tflite_runtime（Linux / ARM 边缘设备）或 tensorflow.lite（Windows）。

    说明：
        - 兼容 NHWC (1,H,W,3) 与 NCHW (1,3,H,W) 两种输入布局；
        - 兼容 float32 与 uint8/int8 量化输入；
        - 输出为量化张量时自动按 scale/offset 反量化。
    """

    def __init__(self, model_cfg: Dict):
        super().__init__(model_cfg)
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                from ai_edge_litert.interpreter import Interpreter  # tflite-runtime 继任者
            except ImportError:
                try:
                    from tensorflow.lite.python.interpreter import Interpreter
                except ImportError:
                    raise ImportError(
                        "TFLite 后端需要 tflite_runtime / ai-edge-litert / tensorflow，"
                        "请先安装（Linux/ARM 推荐 pip install tflite-runtime，"
                        "Windows 推荐 pip install ai-edge-litert）")

        path = model_cfg["tflite"]["model_path"]
        if not Path(path).exists():
            raise FileNotFoundError(
                f"未找到 TFLite 模型: {path}。请先运行 scripts/train_pose.py --export-tflite")
        self.interpreter = Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.in_details = self.interpreter.get_input_details()[0]
        self.out_details = self.interpreter.get_output_details()[0]
        self.in_shape = self.in_details["shape"]
        self.in_dtype = self.in_details["dtype"]

    def predict(self, frame: np.ndarray) -> List[Dict]:
        size = (int(max(self.in_shape[1:3]))
                if len(self.in_shape) == 4 else self.input_size)
        img, scale, pad_w, pad_h = _letterbox(frame, size)

        # 输入布局：NCHW -> 转置；NHWC -> 保持
        if len(self.in_shape) == 4 and self.in_shape[1] == 3:
            blob = img.transpose(2, 0, 1)[None, ...]
        else:
            blob = img[None, ...]
        qscale, qoffset = self.in_details.get("quantization", (0.0, 0))
        if self.in_dtype == np.uint8:
            blob = blob.astype(np.uint8)
        elif self.in_dtype == np.int8:
            # INT8 模型：输入量化参数基于 [0,1] 归一化浮点，先归一化再量化
            blob = np.clip(
                (blob.astype(np.float32) / 255.0) / qscale + qoffset,
                -128, 127).astype(np.int8)
        else:
            blob = blob.astype(np.float32) / 255.0     # 浮点模型归一化

        self.interpreter.set_tensor(self.in_details["index"], blob)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.out_details["index"])

        # 量化输出反量化（float32 输出的 quantization 为 (0.0, 0)，跳过）
        q = self.out_details.get("quantization", (None, None))
        qscale, qoffset = q if isinstance(q, (tuple, list)) else (None, None)
        if qscale not in (None, 0.0):
            out = (out.astype(np.float32) - float(qoffset)) * float(qscale)

        out = np.squeeze(out)
        if out.shape[0] > out.shape[1]:                # (N, C) -> (C, N)
            out = out.T
        return _parse_raw_predictions(
            out, self.num_keypoints, self.conf_thres,
            self.kpt_conf_thres, scale, pad_w, pad_h)


class MockDetector(GaugeDetector):
    """调试后端：从配置读取归一化关键点，换算成原图像素坐标返回。

    仅用于“无权重联调”读数和可视化链路，不产生真实检测结果。
    """

    def __init__(self, model_cfg: Dict):
        super().__init__(model_cfg)
        kpts = model_cfg.get("mock", {}).get("keypoints")
        if not kpts or len(kpts) != 4:
            raise ValueError(
                "mock 后端需要在 config/gauge.yaml 的 model.mock.keypoints 提供"
                " 4 个关键点（顺序 [min, max, tip, center]，归一化坐标 0~1）")
        self.mock_kpts = [[float(x), float(y)] for x, y in kpts]

    def predict(self, frame: np.ndarray) -> List[Dict]:
        h, w = frame.shape[:2]
        kpts = [
            [x * w, y * h] if (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) else [x, y]
            for x, y in self.mock_kpts
        ]
        return [{
            "bbox": (0.0, 0.0, float(w), float(h)),
            "keypoints": kpts,
            "conf": 1.0,
            "kpt_conf": [1.0] * self.num_keypoints,
        }]


def create_detector(config: Dict) -> GaugeDetector:
    """工厂函数：按 config["model"]["backend"] 创建对应后端实例。"""
    model_cfg = config["model"]
    backend = str(model_cfg.get("backend", "mock")).lower()
    registry: Dict[str, type] = {
        "ultralytics": UltralyticsDetector,
        "onnx": ONNXDetector,
        "tflite": TFLiteDetector,
        "mock": MockDetector,
    }
    if backend not in registry:
        raise ValueError(f"未知 backend: {backend}（可选 {list(registry)}）")
    return registry[backend](model_cfg)
