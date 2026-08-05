# 模型目录

训练 / 导出产物统一放在本目录，`config/gauge.yaml` 中的默认路径与之一致：

| 文件 | 后端 | 说明 |
| --- | --- | --- |
| `gauge_pose_e30.pt` | `ultralytics` | 30 epoch 版本 PyTorch 权重（当前默认，保留用于对比/回退） |
| `gauge_pose_e30.onnx` | `onnx` | 30 epoch 版本 ONNX（当前默认，`cv2.dnn` 加载） |
| `gauge_pose.tflite` | `tflite` | TFLite（`tflite_runtime` 加载），RK3588 / Jetson / RPi 首选 |

> 下一轮 120 epoch 训练完成后，`scripts/train_pose.py` 会把新的权重写入
> `gauge_pose.pt` / `gauge_pose.onnx`（不带后缀），届时把 `config/gauge.yaml`
> 的模型路径切回即可，两个版本可随时对比。

## 生成命令（项目根目录）

```bash
# 完整流程：OpenCV 校验 -> 训练 -> 导出 ONNX
python scripts/train_pose.py --epochs 120 --batch 16 --device 0 --export-onnx

# 额外导出 TFLite（需 tensorflow）
python scripts/train_pose.py --epochs 120 --batch 16 --device 0 --export-onnx --export-tflite
```

关键点顺序：导出模型输出顺序与训练一致 `[min, max, tip, center]`（索引 0/1/2/3），
`src/detector.py` 不做重排，下游直接按此顺序使用。
