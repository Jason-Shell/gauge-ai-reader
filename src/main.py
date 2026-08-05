# -*- coding: utf-8 -*-
"""main.py —— 程序入口：argparse 解析、图像/视频/摄像头读取、推理主循环。

用法示例（在项目根目录执行）：
    # 1) 处理单张图片（默认配置，mock 后端，无权重联调）
    python src/main.py --source "gauge_sim_3 dataset/0.jpg"

    # 2) 批量处理整个目录下的图片（headless 服务器环境）
    python src/main.py --source "gauge_sim_3 dataset" --headless

    # 3) 摄像头实时推理
    python src/main.py --source 0

    # 4) 视频文件推理并保存结果
    python src/main.py --source video.mp4 --out Results/out.mp4 --headless

    # 5) 切换推理后端（覆盖 config/gauge.yaml 中的 model.backend）
    python src/main.py --source meter.jpg --backend onnx

OpenCV 职责仅限：摄像头/视频读取、图像预处理、关键点连线绘制、文本显示；
检测与关键点来自深度学习模型（YOLOv8-Pose）；传统方法（径向扫描 /
椭圆拟合）仅作为可选精修层（refiner.py），失败自动回退 DL 结果。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 兼容两种启动方式：python src/main.py 与 python -m src.main
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import yaml

from reader import GaugeReader, Reading
from mqtt_publisher import MqttPublisher
from visualizer import draw_readings, format_reading_value

PROJECT_DIR = SRC_DIR.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

# ============ 读数凑整控制（直接改这里即可切换，无需改配置/命令行） ============
# ROUND_ENABLED: True  = 开启凑整，读数近似到 step 的倍数
#                False = 关闭凑整，保持原始精度
# ROUND_STEP: 1=整数，10=整十，5=五的倍数，0.5=半格（仅 ROUND_ENABLED=True 时生效）
ROUND_ENABLED = False
ROUND_STEP = 10


def resolve_path(path: str) -> Path:
    """相对路径统一基于项目根目录解析，不依赖当前工作目录。"""
    p = Path(str(path).strip())
    return p if p.is_absolute() else PROJECT_DIR / p


def load_config(path: Path) -> Dict:
    """加载并校验配置，缺关键节时给出友好报错。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件格式错误（应为 YAML 映射）: {path}")
    for section in ("gauge", "model", "io"):
        if section not in config:
            raise ValueError(f"配置缺少 {section} 节: {path}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="指针式仪表自动读数（YOLOv8-Pose 关键点 + 双量程解算）")
    parser.add_argument("--config", default="config/gauge.yaml",
                        help="配置文件路径（默认 config/gauge.yaml）")
    parser.add_argument("--source", default=None,
                        help="图片/目录/视频路径、RTSP 地址或摄像头编号（如 0）")
    parser.add_argument("--out", default=None,
                        help="输出路径：图片保存为文件，目录批量保存为目录，视频保存为 mp4")
    parser.add_argument("--backend", default=None,
                        choices=["ultralytics", "onnx", "tflite", "mock"],
                        help="覆盖配置中的推理后端")
    parser.add_argument("--conf", type=float, default=None,
                        help="覆盖检测置信度阈值")
    parser.add_argument("--headless", action="store_true",
                        help="无 GUI 环境，不弹窗显示")
    parser.add_argument("--show", action="store_true",
                        help="强制弹窗显示（默认按配置文件 io.show）")
    parser.add_argument("--mqtt", action="store_true",
                        help="强制开启 MQTT 上传（默认按配置文件 mqtt.enabled）")
    parser.add_argument("--round-to", type=float, default=None,
                        help="开启读数凑整并指定步长（临时覆盖 main.py 的 ROUND_STEP）："
                             "1=整数，10=整十，5=五的倍数")
    parser.add_argument("--no-round", action="store_true",
                        help="强制关闭读数凑整（临时覆盖 main.py 的 ROUND_ENABLED）")
    return parser.parse_args()


def _print_readings(readings: List[Reading]) -> None:
    """终端输出单帧/单图所有表盘的读数摘要。"""
    for i, r in enumerate(readings):
        if r.error:
            print(f"  [{i}] 解算失败: {r.error}")
            continue
        print(
            f"  [{i}] bar: {format_reading_value(r.primary_value):>7} | "
            f"psi: {format_reading_value(r.secondary_value):>8} | "
            f"ratio: {r.ratio * 100:5.1f}% | "
            f"conf: {r.conf:.2f}"
            + (f" | refine: {r.refine_conf:.2f}" if r.refine_used else "")
            + (f" | refine-fail: {r.refine_error[:40]}" if r.refine_error else ""))


def _annotate(reader: GaugeReader, frame, gauge_cfg: Dict):
    """单帧推理 + 绘制，返回 (标注图, 读数列表)。"""
    readings = reader.read_frame(frame)
    annotated = frame.copy()  # 不在原帧上绘制，避免污染后续处理
    draw_readings(annotated, readings, gauge_cfg)
    return annotated, readings


def process_image_file(reader: GaugeReader, img_path: Path, out_path: Optional[Path],
                       gauge_cfg: Dict, show: bool, save: bool,
                       publisher: Optional[MqttPublisher] = None) -> None:
    """处理单张图片。"""
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"[main] 无法读取图片: {img_path}")
        return
    annotated, readings = _annotate(reader, frame, gauge_cfg)
    print(f"[main] {img_path.name}:")
    _print_readings(readings)
    if publisher is not None:
        publisher.maybe_publish(str(img_path), readings)
    if save and out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"[main] 已保存: {out_path}")
    if show:
        cv2.imshow("gauge-reading", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_directory(reader: GaugeReader, src_dir: Path, out_dir: Path,
                      gauge_cfg: Dict, show: bool, save: bool,
                      publisher: Optional[MqttPublisher] = None) -> None:
    """批量处理目录下的全部图片。"""
    images = sorted(
        p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"[main] 目录中没有图片: {src_dir}")
        return
    print(f"[main] 批量处理 {len(images)} 张图片: {src_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in images:
        process_image_file(reader, p, out_dir / p.name, gauge_cfg,
                           show=False, save=save, publisher=publisher)
    if show:
        cv2.destroyAllWindows()


def process_stream(reader: GaugeReader, source, out_path: Optional[Path],
                   gauge_cfg: Dict, show: bool, save: bool,
                   publisher: Optional[MqttPublisher] = None) -> None:
    """视频文件 / RTSP / 摄像头实时推理主循环。"""
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")

    writer = None
    if save and out_path is not None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 未指定具体视频文件时，自动生成 <名称>_result.mp4
        if out_path.suffix.lower() not in VIDEO_EXTS:
            stem = Path(str(source)).stem or "camera"
            out_path = out_path / f"{stem}_result.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        print(f"[main] 输出视频: {out_path}")

    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, readings = _annotate(reader, frame, gauge_cfg)
            if publisher is not None:
                publisher.maybe_publish(str(source), readings)
            if frame_id % 30 == 0:
                print(f"[main] frame {frame_id}:")
                _print_readings(readings)
            if writer is not None:
                writer.write(annotated)
            if show:
                cv2.imshow("gauge-reading", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_id += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))

    # 命令行覆盖配置
    if args.backend:
        config["model"]["backend"] = args.backend
    if args.conf is not None:
        config["model"]["conf_thres"] = float(args.conf)
    if args.mqtt:
        config.setdefault("mqtt", {})["enabled"] = True
    # 读数凑整：优先级 命令行参数 > main.py 顶部常量（ROUND_ENABLED / ROUND_STEP）
    rounding = config["gauge"].setdefault("rounding", {})
    if args.round_to is not None:
        rounding["enabled"] = True
        rounding["step"] = float(args.round_to)
    elif args.no_round:
        rounding["enabled"] = False
    else:
        rounding["enabled"] = bool(ROUND_ENABLED)
        rounding["step"] = float(ROUND_STEP)

    gauge_cfg = config["gauge"]
    io_cfg = config["io"]
    show = args.show or (bool(io_cfg.get("show", True)) and not args.headless)
    save = bool(io_cfg.get("save", True)) or args.out is not None

    reader = GaugeReader(config)
    publisher = MqttPublisher(config.get("mqtt"))
    print(f"[main] 后端: {reader.detector.backend}")
    if rounding["enabled"]:
        print(f"[main] 凑整: 开启（步长 {rounding['step']:g}）")
    else:
        print("[main] 凑整: 关闭（保持原始精度）")

    source = args.source or io_cfg.get("input_image", "meter.jpg")
    out_path = args.out or str(resolve_path(io_cfg.get("out_dir", "Results")))

    try:
        # 摄像头编号
        if str(source).isdigit():
            process_stream(reader, source, resolve_path(out_path),
                           gauge_cfg, show, save, publisher)
            return

        src_path = resolve_path(str(source))
        if src_path.is_dir():
            process_directory(reader, src_path, resolve_path(out_path),
                              gauge_cfg, show, save, publisher)
            return

        if not src_path.exists():
            raise FileNotFoundError(f"输入不存在: {src_path}")

        if src_path.suffix.lower() in IMAGE_EXTS:
            # 图片输出：--out 是文件则写文件，否则写默认目录
            out_file = resolve_path(out_path)
            if out_file.suffix.lower() not in IMAGE_EXTS:
                out_file = out_file / src_path.name
            process_image_file(reader, src_path, out_file, gauge_cfg,
                               show, save, publisher)
            return

        # 视频 / RTSP
        process_stream(reader, src_path, resolve_path(out_path),
                       gauge_cfg, show, save, publisher)
    finally:
        publisher.close()


if __name__ == "__main__":
    main()
