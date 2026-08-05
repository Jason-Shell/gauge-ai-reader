# 模型目录

训练 / 导出产物统一放在本目录，`config/gauge.yaml` 中的默认路径与之一致。

| 文件 | 后端 | 说明 |
| --- | --- | --- |
| `gauge_pose_e120.pt` | `ultralytics` | **当前默认**：120 epoch 版本（config 已指向） |
| `gauge_pose_e120.onnx` | `onnx` | 当前默认 ONNX（`cv2.dnn` 加载） |
| `gauge_pose_e30.pt` | `ultralytics` | 30 epoch 版本，保留用于对比 / 回退 |
| `gauge_pose_e30.onnx` | `onnx` | 30 epoch 版本 ONNX |
| `gauge_pose_e120.tflite` | `tflite` | **当前默认 TFLite**：float32，由 `scripts/onnx_to_tflite.py` 转换 |
| `gauge_pose_e120_int8.tflite` | `tflite` | INT8 失败基线：单输出张量量化坍缩置信度，不用于部署 |

> 状态说明：`config/gauge.yaml` 的 `model.pt/onnx.model_path` 均指向
> `gauge_pose_e120.*`，`model.tflite.model_path` 已指向
> `gauge_pose_e120.tflite`，与上表一致。

## 生成命令（项目根目录）

```powershell
# 完整流程：OpenCV 校验 -> 训练 -> 导出 ONNX
& "D:\JasonXie\Code-OpenCV\Project\.python312\python.exe" scripts\train_pose.py --epochs 120 --batch 16 --device 0 --export-onnx

# TFLite 独立转换（tf_env，不依赖 ultralytics；onnx-tf 需 onnx<=1.15）
& "D:\Anaconda3\envs\tf_env\python.exe" scripts\onnx_to_tflite.py --out models\gauge_pose_e120.tflite

# INT8 全量化（当前 YOLO 导出会量化坍缩，仅供基线对比）
& "D:\Anaconda3\envs\tf_env\python.exe" scripts\onnx_to_tflite.py --out models\gauge_pose_e120.tflite --int8
```

关键点顺序：导出模型输出顺序与训练一致 `[min, max, tip, center]`
（索引 0/1/2/3），`src/detector.py` 不做重排，下游直接按此顺序使用。
