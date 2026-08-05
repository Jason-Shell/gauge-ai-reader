# 项目交接提示词（复制给新的 Codex 对话）

> 用法：把下面整段复制到新的 Codex 对话窗口，即可无缝续接本项目。

---

你是 Codex。请继续维护/开发 `D:\JasonXie\Code-OpenCV\Project` 下的指针式仪表自动读数系统（GitHub 仓库 gauge-ai-reader，分支 main）。开始前先读 `README.md` 和 `firmware/README.md` 了解全貌，然后跑一遍下面的"验证命令"确认基线，再按用户当前目标工作。

## 项目定位

基于 **YOLOv8-Pose 关键点检测 + 双量程角度比例解算**（bar 0~250 / psi 0~3625.9）的指针式压力表自动读数系统。DL 主检 + 传统方法精修（失败自动回退）。已覆盖：数据集校验、训练、ONNX/TFLite 导出、推理、MQTT、端到端精度评估、性能基准、ESP32-S3 传统法固件。

## 环境（重要：用对解释器）

- **主项目 Python**（推理/训练/评估/测试一律用它）：`D:\JasonXie\Code-OpenCV\Project\.python312\python.exe`（3.12.10，含 torch cu121 / ultralytics / cv2 / onnxruntime / paho-mqtt，**无 tensorflow**）
- **TF 环境**（仅 TF 相关工作时用）：`D:\Anaconda3\envs\tf_env\python.exe`（3.10.20 + TensorFlow 2.21，TFLiteConverter 可用；**无 cv2 / ultralytics**）
- 依赖锁：`requirements.lock`；VS Code 默认解释器已指向 tf_env，主项目请用 `.python312`
- **红线**：不要向 `.python312` 安装 tensorflow（用户明确不要重复装）；不要删除 `trash/`（用户保留）；`datasets/` 不入库；`firmware/.pio/`、`Results/`、`runs/` 不入库
- 沙箱限制：`C:\Users\lscm\...` 不可读；`D:\Anaconda3` 与项目目录可读；联网 / pip 安装 / git push 需提权（git push 已有常驻审批；带 `2>&1` 等重定向的命令不会命中审批规则）

## 关键约定

- 关键点顺序固定 `[min, max, tip, center]`，全链路统一，绝不重排
- DL 负责表盘检测与关键点；传统方法（径向扫描 / 椭圆拟合）只允许在 `src/refiner.py` 中、以 DL 结果为先验做数值精修，失败必须回退
- 默认精修配置：**中心椭圆拟合开启**（仿真集 MAE 2.40 → 1.74 bar）；**指针径向扫描默认关闭**（实验性，与中心精修联用会放大分歧）
- `sweep_direction: auto` 的已知局限：两条互补弧分割圆周，指针在量程弧内可正确判定；超量程指针会被按互补弧误读（已在 README 记录，生产建议按表盘型号固定方向）
- 透视归一化（`gauge.perspective`）为实验性功能，默认关闭

## 现有功能与文件

- `src/detector.py`：ultralytics / onnx / tflite / mock 多后端（工厂模式）
- `src/refiner.py`：径向扫描指针精修 + 椭圆拟合中心精修 + 共识门控
- `src/geometry.py`：角度 / 比例 / 双量程 / 实验性透视归一化
- `src/reader.py`、`src/main.py`、`src/mqtt_publisher.py`、`src/visualizer.py`
- `scripts/train_pose.py`、`scripts/eval_reading.py`（端到端读数精度）、`scripts/benchmark.py`（延迟/FPS）、`scripts/tf_smoke.py`（tf_env 工具链验证）
- `tests/`：35 个 unittest（geometry / refiner / reader 端到端）
- `firmware/`：ESP32-S3 传统法读数固件 v1（PlatformIO；OV2640 针脚沿用旧工程 `D:\JasonXie\PlatformIO\Projects\my_first_esp32s3`；径向扫描 C++ 移植已通过主机 g++ 回归测试；完整编译需用户 VS Code + PlatformIO）

## 验证命令（新对话先跑）

```powershell
& "D:\JasonXie\Code-OpenCV\Project\.python312\python.exe" -m unittest discover -s tests
& "D:\JasonXie\Code-OpenCV\Project\.python312\python.exe" scripts\eval_reading.py --source "gauge_sim_3 dataset" --backend onnx
```

预期：35 个测试全过；仿真集 MAE ≈ 1.74 bar（DL + 中心精修）。

## 当前状态与续接点

- **传统法固件 v1（STA 模式，串口配网）已推送**，待用户操作：VS Code 打开 `firmware/` → Upload 烧录 → 串口 `WIFI SET <ssid> <pass>` 配网（密码存 NVS）→ 串口标定（`CAL CENTER AUTO` → `CAL MIN` → `CAL MAX` → `CAL SWEEP cw|ccw` → `CAL SAVE`）→ 手机/电脑连同一局域网 → 浏览器打开串口日志打印的 IP 或 `http://gauge.local/` 看读数
- **路线 B（待用户实拍照片后启动）**：用 tf_env（Keras）训练微型关键点/角度回归模型（输入 96~128px，INT8 TFLite <1MB），再集成 TFLite Micro 到 firmware
- **备选**：写 ONNX → SavedModel → TFLite 独立转换脚本（不依赖 ultralytics），把现有 YOLO 模型导出成 TFLite 作对比基线
- **待用户反馈**：firmware 编译/烧录结果、`/capture` 实拍照片

## 工作原则

- 改动前先跑 unittest；改动后跑 `eval_reading.py` 确认读数精度不回归
- 编辑文件用 apply_patch，不要用 cat 等 shell 写文件
- 涉及删除、联网、装包、推送到外部前先向用户确认（git push 除外，已有常驻审批）
- 发现 README 与代码不一致时，按实际程序修改文档并提示用户
- 单次改动尽量小步提交，commit message 写清"为什么"
