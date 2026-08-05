# 模型目录

训练 / 导出产物统一放在本目录，`config/gauge.yaml` 中的默认路径与之一致。

| 文件 | 后端 | 说明 |
| --- | --- | --- |
| `gauge_pose_e120.pt` | `ultralytics` | **当前默认**：120 epoch 版本（config 已指向） |
| `gauge_pose_e120.onnx` | `onnx` | 当前默认 ONNX（`cv2.dnn` 加载） |
| `gauge_pose_e30.pt` | `ultralytics` | 30 epoch 版本，保留用于对比 / 回退 |
| `gauge_pose_e30.onnx` | `onnx` | 30 epoch 版本 ONNX |
| `gauge_pose.tflite` | `tflite` | **尚未导出**：需先运行 `--export-tflite`（需 tensorflow） |

> 状态说明：`config/gauge.yaml` 的 `model.pt/onnx.model_path` 均指向
> `gauge_pose_e120.*`，与上表一致；TFLite 模型缺失，切换到 `backend: tflite`
> 会报错，请先按下方命令导出。

## 生成命令（项目根目录）

```powershell
# 完整流程：OpenCV 校验 -> 训练 -> 导出 ONNX
& "D:\JasonXie\Code-OpenCV\Project\.python312\python.exe" scripts\train_pose.py --epochs 120 --batch 16 --device 0 --export-onnx

# 额外导出 TFLite（需 tensorflow）
& "D:\JasonXie\Code-OpenCV\Project\.python312\python.exe" scripts\train_pose.py --epochs 120 --batch 16 --device 0 --export-onnx --export-tflite
```

关键点顺序：导出模型输出顺序与训练一致 `[min, max, tip, center]`
（索引 0/1/2/3），`src/detector.py` 不做重排，下游直接按此顺序使用。
