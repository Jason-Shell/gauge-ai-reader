# -*- coding: utf-8 -*-
"""onnx_to_tflite.py —— 独立 ONNX -> SavedModel -> TFLite 转换脚本（不依赖 ultralytics）。

用 tf_env 运行（TF 2.21 / Python 3.10，TFLiteConverter 可用）：

    & "D:\\Anaconda3\\envs\\tf_env\\python.exe" scripts\\onnx_to_tflite.py \
        --onnx models/gauge_pose_e120.onnx --out models/gauge_pose_e120.tflite

    # INT8 全量化（路线 B 的 <1MB 微型模型走同样接口，此处先给现有 YOLO 做尺寸/精度基线）：
    & "D:\\Anaconda3\\envs\\tf_env\\python.exe" scripts\\onnx_to_tflite.py \
        --onnx models/gauge_pose_e120.onnx --out models/gauge_pose_e120_int8.tflite --int8

转换链路：onnx -> onnx-tf(TensorflowRep) -> tf.saved_model(export_graph) -> TFLiteConverter。
转换完成后自动用 Interpreter 做一次推理回环自检，并给出与 onnx-tf 参考输出的最大偏差。

为什么需要“空桩”：
    onnx-tf 1.10.0 已停止维护，其 Bernoulli / Hardmax 算子 handler 在模块导入期
    强制 import tensorflow_probability / tensorflow_addons；而 TFA 0.22 与 TF 2.21
    （Keras 3）不兼容、TFP 是重型依赖。本项目模型（YOLOv8-Pose 子图）只用
    Conv / Mul / Sigmoid / Resize / Concat / Split 等标准算子，运行期不会触达
    这两个依赖，故在导入 onnx_tf 前用空命名空间模块替代。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import types
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


def _install_onnx_tf_stubs() -> None:
    """在导入 onnx_tf 前注册 TFP / TFA 空桩（见模块 docstring）。"""
    for name in ("tensorflow_probability", "tensorflow_probability.distributions"):
        mod = types.ModuleType(name)
        sys.modules[name] = mod

    tfa = types.ModuleType("tensorflow_addons")
    seq = types.ModuleType("tensorflow_addons.seq2seq")

    def _hardmax(x, *args, **kwargs):  # 仅满足导入期装饰器，运行期不会调用
        return x

    seq.hardmax = _hardmax
    tfa.seq2seq = seq
    sys.modules["tensorflow_addons"] = tfa
    sys.modules["tensorflow_addons.seq2seq"] = seq


def _load_calib_image(path: str, size: int = 640) -> np.ndarray:
    """读取 JPEG -> letterbox 到 size x size -> [0,1] 归一化，返回 NCHW float32。"""
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # uint8 -> [0,1]
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    scale = tf.minimum(tf.cast(size, tf.float32) / h, tf.cast(size, tf.float32) / w)
    new_h = tf.cast(tf.math.round(h * scale), tf.int32)
    new_w = tf.cast(tf.math.round(w * scale), tf.int32)
    img = tf.image.resize(img, (new_h, new_w), method="bilinear")
    pad_h = (size - new_h) // 2
    pad_w = (size - new_w) // 2
    img = tf.image.pad_to_bounding_box(img, pad_h, pad_w, size, size)
    return img.numpy()[None, ...].transpose(0, 3, 1, 2)  # (1,3,H,W) [0,1]


def _collect_calib_paths(calib_dir: str, calib_n: int) -> list[Path]:
    root = Path(calib_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"校准图目录不存在: {root}")
    paths = sorted(
        p for suffix in ("*.jpg", "*.jpeg", "*.png") for p in root.glob(suffix)
    )
    if not paths:
        raise FileNotFoundError(f"校准图目录中没有 jpg/jpeg/png: {root}")
    return paths[:calib_n]


def _make_representative_gen(calib_paths: list[Path], size: int):
    def gen():
        for p in calib_paths:
            yield [_load_calib_image(str(p), size=size)]
    return gen


def _interpreter_run(tflite_path: str, x: np.ndarray) -> np.ndarray:
    """用 TFLite Interpreter 执行单次推理（优先 LiteRT，回退 tf.lite）。"""
    try:
        from ai_edge_litert.interpreter import Interpreter  # TF 2.20+ 新接口
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

    interp = Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    if in_d["dtype"] != np.float32:
        # INT8 模型：按量化参数把 [0,1] 浮点输入转成 int8 张量
        qscale, qoffset = in_d["quantization"]
        feed = np.clip(x / qscale + qoffset, -128, 127).astype(np.int8)
    else:
        feed = x.astype(np.float32)
    interp.set_tensor(in_d["index"], feed)
    interp.invoke()
    out = interp.get_tensor(out_d["index"])
    # float32 输出的 quantization 为 (0.0, 0)，须跳过，否则输出被乘成 0
    q = out_d.get("quantization", (None, None))
    qscale, qoffset = q if isinstance(q, (tuple, list)) else (None, None)
    if qscale not in (None, 0.0):
        out = (out.astype(np.float32) - float(qoffset)) * float(qscale)
    return out


def _convert(onnx_path: Path, out_path: Path, work_dir: Path,
             int8: bool, calib_dir: str, calib_n: int,
             keep_saved_model: bool, size: int) -> None:
    import onnx
    from onnx_tf.backend import prepare

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    saved_model_dir = work_dir / "saved_model"

    model = onnx.load(str(onnx_path))
    opset = [(o.domain, o.version) for o in model.opset_import]
    print(f"[onnx] 加载 {onnx_path.name}（{onnx_path.stat().st_size / 1e6:.1f} MB），opset={opset}")

    tf_rep = prepare(model)
    print("[onnx-tf] 图转换完成")

    tf_rep.export_graph(str(saved_model_dir))
    print(f"[saved_model] 导出完成: {saved_model_dir}")

    # ---- float32 ----
    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite = conv.convert()
    out_path.write_bytes(tflite)
    print(f"[tflite] float32 -> {out_path} ({len(tflite) / 1e6:.2f} MB)")

    # ---- INT8（可选）----
    if int8:
        calib_paths = _collect_calib_paths(calib_dir, calib_n)
        print(f"[tflite] INT8 校准图 {len(calib_paths)} 张（来自 {calib_dir}）")
        conv8 = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        conv8.optimizations = [tf.lite.Optimize.DEFAULT]
        conv8.representative_dataset = _make_representative_gen(calib_paths, size)
        conv8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv8.inference_input_type = tf.int8
        conv8.inference_output_type = tf.int8
        tflite8 = conv8.convert()
        int8_path = out_path.with_name(out_path.stem + "_int8.tflite")
        int8_path.write_bytes(tflite8)
        print(f"[tflite] int8 全量化 -> {int8_path} ({len(tflite8) / 1e6:.2f} MB)")

    # ---- 推理回环自检 ----
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, size, size)).astype(np.float32) * 0.25 + 0.5
    y_ref = tf_rep.run({"images": x})["output0"]
    y_tfl = _interpreter_run(str(out_path), x)
    print(f"[check] float32 最大偏差 vs onnx-tf 参考: {float(np.abs(y_tfl - y_ref).max()):.6f}")
    if int8:
        y_i8 = _interpreter_run(str(int8_path), x)
        diff_i8 = float(np.abs(y_i8 - y_ref).max())
        print(f"[check] int8    最大偏差 vs onnx-tf 参考: {diff_i8:.6f}")
        if diff_i8 > 1.0:
            print("[warn] INT8 输出偏差过大：单输出张量同时含宽动态范围坐标(0~700)与"
                  "[0,1] 置信度时，逐张量量化会把置信度压成 0（本模型实测全样本"
                  "无检测）。该基线不适合直接部署，建议改用 float32 / float16，"
                  "或拆分为多输出张量后再量化。")

    if not keep_saved_model:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"[clean] 已清理中间目录: {work_dir}（--keep-saved-model 可保留）")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ONNX -> SavedModel -> TFLite 独立转换（用 tf_env 运行）")
    parser.add_argument("--onnx", default="models/gauge_pose_e120.onnx",
                        help="输入 ONNX 模型路径（默认 models/gauge_pose_e120.onnx）")
    parser.add_argument("--out", default=None,
                        help="输出 TFLite 路径；--int8 时默认追加 _int8 后缀的文件")
    parser.add_argument("--int8", action="store_true",
                        help="同时导出 INT8 全量化版本（需 --calib-dir 校准图）")
    parser.add_argument("--calib-dir", default="gauge_sim_3 dataset",
                        help="INT8 校准图目录（默认仿真集）")
    parser.add_argument("--calib-n", type=int, default=20,
                        help="INT8 校准图数量（默认 20）")
    parser.add_argument("--input-size", type=int, default=640,
                        help="模型输入边长（默认 640，须与 ONNX 输入一致）")
    parser.add_argument("--keep-saved-model", action="store_true",
                        help="保留 SavedModel 中间产物（默认转换后清理）")
    args = parser.parse_args()

    print(f"TensorFlow {tf.__version__}")
    _install_onnx_tf_stubs()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX 模型不存在: {onnx_path}")
    out_path = Path(args.out) if args.out else (
        onnx_path.with_suffix(".tflite") if not args.int8
        else onnx_path.with_name(onnx_path.stem + "_int8.tflite"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path("Results") / "onnx_to_tflite_work"

    _convert(onnx_path, out_path, work_dir, args.int8,
             args.calib_dir, args.calib_n, args.keep_saved_model,
             args.input_size)
    print("done")


if __name__ == "__main__":
    main()
