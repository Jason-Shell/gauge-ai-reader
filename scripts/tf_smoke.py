# -*- coding: utf-8 -*-
"""tf_smoke.py —— tf_env 工具链最小验证（Keras -> TFLite -> 推理回环）。

用 Anaconda 的 tf_env 运行（TF 2.21 / Python 3.10）：
    & "D:\\Anaconda3\\envs\\tf_env\\python.exe" scripts\\tf_smoke.py

验证内容：Keras 建模/训练 -> float32 TFLite 导出 -> INT8 量化导出 ->
tf.lite.Interpreter 加载推理。输出写入 Results/tf_smoke/（不入库）。
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


def main() -> None:
    print("TensorFlow", tf.__version__)

    # 小型回归模型：8 维输入 -> 1 维输出（模拟“关键点 -> 读数”的最小链路）
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(8,)),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    x = np.random.rand(64, 8).astype(np.float32)
    y = np.random.rand(64, 1).astype(np.float32)
    model.fit(x, y, epochs=2, verbose=0)
    print("Keras train OK")

    out_dir = Path(__file__).resolve().parents[1] / "Results" / "tf_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- float32 转换 ----
    tflite_path = out_dir / "model.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite = converter.convert()
    tflite_path.write_bytes(tflite)
    print(f"TFLite saved: {tflite_path} ({len(tflite) / 1024:.1f} KB)")

    # ---- Interpreter 推理回环 ----
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    sample = np.random.rand(1, 8).astype(np.float32)
    interpreter.set_tensor(inp["index"], sample)
    interpreter.invoke()
    result = interpreter.get_tensor(out["index"])
    print(f"float32 inference OK -> {result[0][0]:.5f}")

    # ---- INT8 量化（路线 B 上 ESP32 必须）----
    def representative():
        for _ in range(20):
            yield [np.random.rand(1, 8).astype(np.float32)]

    converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter2.optimizations = [tf.lite.Optimize.DEFAULT]
    converter2.representative_dataset = representative
    converter2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter2.inference_input_type = tf.int8
    converter2.inference_output_type = tf.int8
    tflite_int8 = converter2.convert()
    int8_path = out_dir / "model_int8.tflite"
    int8_path.write_bytes(tflite_int8)
    print(f"INT8 TFLite saved: {int8_path} ({len(tflite_int8) / 1024:.1f} KB)")

    interpreter8 = tf.lite.Interpreter(model_path=str(int8_path))
    interpreter8.allocate_tensors()
    inp8 = interpreter8.get_input_details()[0]
    out8 = interpreter8.get_output_details()[0]
    interpreter8.set_tensor(inp8["index"], sample.astype(np.int8))
    interpreter8.invoke()
    print(f"INT8 inference OK -> {interpreter8.get_tensor(out8['index'])[0][0]}")

    print("tf_env toolchain OK")


if __name__ == "__main__":
    main()
