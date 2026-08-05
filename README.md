# 指针式仪表自动读数系统（YOLOv8-Pose 关键点 + 双量程解算）

基于 **Top-Down 关键点检测** 的指针式仪表自动读数系统，彻底移除传统
OpenCV 图像处理（Hough 圆、连通域、轮廓提取等），覆盖
**数据集校验 -> 训练 -> 边缘端推理部署** 的完整生命周期。

目标仪表：WIKA 双刻度表盘（主刻度 `bar`，副刻度 `psi`）。

## 关键点定义（严格顺序，全链路统一）

| 索引 | 名称 | 含义 |
| --- | --- | --- |
| 0 | `min` | 刻度起点端（最小值端） |
| 1 | `max` | 刻度终点端（最大值端） |
| 2 | `tip` | 指针尖端 |
| 3 | `center` | 表盘中心 |

## 目录结构

```text
project/
├── datasets/
│   └── Gauge.v1i.yolov8/    # Roboflow 数据集（train/valid/test + data.yaml）
├── scripts/
│   └── train_pose.py        # YAML 修复 + OpenCV 数据校验 + 训练 + ONNX/TFLite 导出
├── models/                  # 存放 .pt / .onnx / .tflite
├── src/
│   ├── detector.py          # GaugeDetector 抽象基类 + ultralytics/onnx/tflite/mock 后端
│   ├── reader.py            # 推理编排：检测 -> 角度比例 -> 双量程映射
│   ├── geometry.py          # 纯数学：atan2 角度、跨 360 度比例、数值映射
│   ├── visualizer.py        # OpenCV 绘制关键点连线与双量程读数
│   └── main.py              # argparse + 图片/视频/摄像头推理主循环
├── config/
│   └── gauge.yaml           # 双量程配置（bar 0~250 / psi 0~3625.9）
└── requirements.txt
```

## 安装依赖

```bash
pip install -r requirements.txt
# Linux/ARM 边缘设备额外安装：pip install tflite-runtime
```

## 1. 数据集可视化校验（传统 OpenCV）

训练前用传统 OpenCV 读取 txt 标签、反归一化并绘制检测框与 4 个关键点，
人工确认标注顺序与归一化是否正确：

```bash
python scripts/train_pose.py --validate-only --limit 20
```

校验图输出到 `runs/validation/`，并生成 `validation_montage.jpg` 网格图。

## 2. 训练与导出

```bash
# 校验 + 训练 + 导出 ONNX（推荐先跑校验，确认无误后训练）
python scripts/train_pose.py --epochs 120 --batch -1 --device 0 --export-onnx

# 同时导出 TFLite（需 tensorflow）
python scripts/train_pose.py --epochs 120 --batch -1 --device 0 --export-onnx --export-tflite
```

训练产物：`runs/pose/gauge/weights/best.pt`，并自动复制为
`models/gauge_pose.pt` / `models/gauge_pose.onnx` / `models/gauge_pose.tflite`。

显存不足时（如 8GB 笔记本 GPU）：

```bash
# batch=-1 会自动探测并选择当前显存能装下的最大 batch
python scripts/train_pose.py --epochs 120 --batch -1 --device 0

# 或手动降低 batch 与输入尺寸（512 比 640 省约一半显存）
python scripts/train_pose.py --epochs 120 --batch 8 --imgsz 512 --device 0
```

## 3. 推理

先在 `config/gauge.yaml` 中把 `model.backend` 切换为真实后端
（`ultralytics` / `onnx` / `tflite`；未训练时可先用 `mock` 联调链路）。

```bash
# 单张图片
python src/main.py --source "gauge_sim_3 dataset/0.jpg"

# 批量目录（headless 服务器）
python src/main.py --source "gauge_sim_3 dataset" --headless

# ONNX 后端
python src/main.py --source meter.jpg --backend onnx

# 摄像头实时推理
python src/main.py --source 0

# 视频文件并保存结果
python src/main.py --source video.mp4 --out Results/out.mp4 --headless
```

终端输出示例：

```text
[main] 后端: mock
[main] 0.jpg:
  [0] bar:    44.2 | psi:    641.2 | ratio:  17.7% | conf: 1.00
```

## 核心算法

1. 以 `center` 为原点，用 `math.atan2()` 计算各点绝对角度并归一化到 `[0, 360)`；
2. 比例计算（自动处理跨 360 度边界与防除零）：

```text
ratio = (Angle_tip - Angle_min) / (Angle_max - Angle_min)   # 限制在 [0, 1]
```

3. 双量程映射：

```text
value = min_value + ratio * (max_value - min_value)
主量程：0.0 ~ 250.0 bar
副量程：0.0 ~ 3625.9 psi
```

## 设计红线

- 推理阶段（`src/`）OpenCV 仅用于摄像头读取、预处理、关键点绘制、文本显示；
  **禁止** HoughCircles / findContours / Canny 等传统轮廓方法；
- 关键点命名严格使用 `min` / `max`（起点端 / 终点端）；
- 训练阶段（`scripts/train_pose.py`）使用传统 OpenCV 做数据集可视化校验。

## 性能建议（边缘部署）

- 部署首选 ONNX（`cv2.dnn`）或 TFLite 后端，避免 PyTorch 运行时开销；
- OpenCV 开启 CUDA 编译时，可在 `config/gauge.yaml` 中设置
  `model.onnx.prefer_cuda: true`；
- 关键点后处理使用 numpy 向量化，无逐锚点 Python 循环。

## MQTT 读数上传（可选）

默认关闭，开启后每张图片 / 每 `interval_sec` 秒向 broker 发布一条 JSON：

```bash
# 方式一：改 config/gauge.yaml 的 mqtt.enabled = true
# 方式二：命令行临时开启
python src/main.py --source 0 --mqtt
```

消息格式：

```text
topic: gauge/reading
{"source": "...", "ts": "...", "readings": [{"index": 0, "bar": 50.5,
 "psi": 732.4, "ratio": 0.202, "conf": 1.0, "keypoints": [[...], ...]}]}
```

未安装 `paho-mqtt` 或 broker 不可达时，程序自动禁用 MQTT 并继续推理。

## 读数凑整（可选）

默认关闭，读数保留原始精度；开启后按指定步长近似到最近的倍数：

```bash
# 近似到整数（101.8 -> 102）
python src/main.py --source meter.jpg --round-to 1

# 近似到整十（154.1 -> 150，199.6 -> 200）
python src/main.py --source meter.jpg --round-to 10

# 近似到五的倍数
python src/main.py --source meter.jpg --round-to 5
```

开关默认放在 `src/main.py` 顶部，改一处即可切换：

```python
# src/main.py 顶部
ROUND_ENABLED = True     # True=开启凑整，False=关闭（保持原始精度）
ROUND_STEP = 10          # 1=整数，10=整十，5=五的倍数，0.5=半格
```

`--round-to <步长>` 可临时开启，`--no-round` 可临时关闭（均覆盖 main.py 常量，不改代码）。
凑整只作用于 bar / psi 读数，`ratio` 和 `conf` 保持原始精度。

## 固定 Python 环境

项目目录内自带一份**自包含 Python 3.12.10 快照**（`.python312/`，从本机基准
环境完整复制，含 CUDA 版 torch 2.5.1+cu121），不依赖用户目录里的安装位置，
在任何工作目录 / 沙箱环境都能直接运行：

```powershell
# 推荐：统一用项目内解释器
& "D:\JasonXie\Code-OpenCV\Project\.python312\python.exe" src\main.py --source meter.jpg --headless

# 训练 / 导出同样适用（自带 CUDA torch）
& "D:\JasonXie\Code-OpenCV\Project\.python312\python.exe" scripts\train_pose.py --epochs 120 --batch -1 --device 0 --export-onnx
```

精确版本锁定见 `requirements.lock`（由 `pip freeze` 生成）。若要重装环境：

```powershell
& "D:\JasonXie\Code-OpenCV\Project\.python312\python.exe" -m pip install -r requirements.lock
```

注意：ultralytics 首次导入会读写用户配置目录；在受限沙箱中可显式指定
`YOLO_CONFIG_DIR` 到项目内（否则会自动回退到项目根下的 `Ultralytics/` 目录）：

```powershell
$env:YOLO_CONFIG_DIR = "D:\JasonXie\Code-OpenCV\Project"
```
