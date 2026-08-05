# -*- coding: utf-8 -*-
"""train_pose.py —— 数据集处理 + 训练 + 导出（YOLOv8-Pose 关键点模型）。

包含三个环节：
    1. YAML 路径修复：把 Roboflow 导出的 data.yaml 改写为 ultralytics 可用的
       绝对 path + 相对 train/val/test + kpt_shape/flip_idx/names；
    2. 传统 OpenCV 数据可视化校验：读取 txt 标签，反归一化后用 cv2.rectangle /
       cv2.circle / cv2.putText 画出检测框与 4 个关键点（min/max/tip/center），
       人工目检标注顺序与归一化是否正确；
    3. 训练 YOLOv8-Pose 并导出 ONNX（可选 TFLite）。

用法示例（在项目根目录执行）：
    # 1) 仅做传统 OpenCV 数据校验（推荐先跑，检查标注顺序与归一化）
    python scripts/train_pose.py --validate-only --limit 20

    # 2) 完整训练（先校验，再训练，导出 ONNX 到 models/）
    #    8GB 显存卡建议 batch=-1（自动探测最大可用 batch）或 --batch 8
    python scripts/train_pose.py --epochs 120 --batch -1 --device 0 --export-onnx

    # 3) 训练并同时导出 ONNX + TFLite
    python scripts/train_pose.py --epochs 120 --batch -1 --device 0 \
        --export-onnx --export-tflite

说明：
    - 首次运行会自动下载预训练权重 yolov8n-pose.pt（需要联网）；
    - 关键点顺序严格固定为 [min, max, tip, center]（索引 0/1/2/3）。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# 把项目根目录加入 sys.path，以便复用 src/geometry.py 中的关键点定义
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.geometry import (  # noqa: E402
    IDX_CENTER, IDX_MAX, IDX_MIN, IDX_TIP, KEYPOINT_NAMES)

# 关键点绘制颜色（BGR），顺序 [min, max, tip, center]
KP_COLORS = [
    (255, 0, 0),      # min：蓝
    (0, 165, 255),    # max：橙
    (0, 0, 255),      # tip：红
    (0, 255, 255),    # center：黄
]

DEFAULT_DATA_YAML = PROJECT_DIR / "datasets" / "Gauge.v1i.yolov8" / "data.yaml"
DEFAULT_MODELS_DIR = PROJECT_DIR / "models"


# ---------------------------------------------------------------------------
# 1. YAML 路径修复
# ---------------------------------------------------------------------------
def fix_data_yaml(data_yaml: Path) -> Path:
    """修复 Roboflow 导出的 data.yaml，保证任何工作目录下都能被 ultralytics 正确加载。

    关键点：
        - path 写绝对路径，train/val/test 用相对 path 的目录；
        - kpt_shape = [4, 3]（4 个关键点，每个 3 个通道 x/y/visible）；
        - flip_idx = [1, 0, 2, 3]：水平翻转时 min <-> max 位置互换，
          tip / center 不变（用于开启 fliplr 增强时的标注一致性）；
        - nc = 1，names = ['gauge']。
    """
    if not data_yaml.exists():
        raise FileNotFoundError(f"未找到数据集配置: {data_yaml}")
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data["path"] = str(data_yaml.parent.resolve())   # 绝对路径，跨目录可用
    data["train"] = "train/images"
    data["val"] = "valid/images"
    data["test"] = "test/images"
    data["kpt_shape"] = [4, 3]
    data["flip_idx"] = [1, 0, 2, 3]
    data["nc"] = 1
    data["names"] = ["gauge"]

    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"[train] 已修复 data.yaml: {data_yaml}")
    return data_yaml


# ---------------------------------------------------------------------------
# 2. 传统 OpenCV 数据可视化校验
# ---------------------------------------------------------------------------
def parse_label_line(line: str, img_w: int, img_h: int) -> Tuple[int, Tuple[int, int, int, int], List[Tuple[Tuple[float, float], int]]]:
    """解析一行 YOLOv8-Pose 标签，并把归一化坐标反归一化为像素坐标。

    标签格式：class cx cy w h  kx0 ky0 v0  kx1 ky1 v1  kx2 ky2 v2  kx3 ky3 v3
    返回：
        class_id,
        (x1, y1, x2, y2) 检测框像素坐标,
        [(关键点像素坐标, 可见性)] * 4，顺序 [min, max, tip, center]
    """
    parts = line.split()
    if len(parts) < 17:
        raise ValueError(f"标签字段不足（期望 17 个，实际 {len(parts)}）：{line}")

    class_id = int(float(parts[0]))
    cx, cy, w, h = (float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4]))
    x1 = int(round((cx - w / 2.0) * img_w))
    y1 = int(round((cy - h / 2.0) * img_h))
    x2 = int(round((cx + w / 2.0) * img_w))
    y2 = int(round((cy + h / 2.0) * img_h))

    keypoints = []
    for i in range(4):
        kx = float(parts[5 + i * 3])
        ky = float(parts[6 + i * 3])
        kv = int(float(parts[7 + i * 3]))   # 可见性：2=可见，1=遮挡，0=未标注
        keypoints.append(((kx * img_w, ky * img_h), kv))
    return class_id, (x1, y1, x2, y2), keypoints


def draw_validation_image(image: np.ndarray, line: str, save_path: Path) -> Dict:
    """用传统 OpenCV 把单条标签可视化到图片上并保存。

    绘制内容：
        - cv2.rectangle：检测框；
        - cv2.circle：4 个关键点（min/max/tip/center，不同颜色）；
        - cv2.line：center 到 min/max/tip 的连线，便于目检顺序；
        - cv2.putText：每个关键点的 "索引:名称" 标签。
    """
    img = image.copy()
    h, w = img.shape[:2]
    class_id, (x1, y1, x2, y2), keypoints = parse_label_line(line, w, h)

    # 检测框
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"class={class_id}", (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # center -> min/max/tip 连线（center 是第 3 号关键点）
    center_pt = keypoints[IDX_CENTER][0]
    for idx, color, thick in ((IDX_MIN, KP_COLORS[IDX_MIN], 2),
                              (IDX_MAX, KP_COLORS[IDX_MAX], 2),
                              (IDX_TIP, KP_COLORS[IDX_TIP], 3)):
        p = keypoints[idx][0]
        cv2.line(img, (int(round(center_pt[0])), int(round(center_pt[1]))),
                 (int(round(p[0])), int(round(p[1]))), color, thick)

    # 4 个关键点圆点 + 文字标签
    for idx, ((px, py), vis) in enumerate(keypoints):
        color = KP_COLORS[idx]
        ix, iy = int(round(px)), int(round(py))
        cv2.circle(img, (ix, iy), 8, color, -1)
        label = f"{idx}:{KEYPOINT_NAMES[idx]}" + ("" if vis > 0 else "(?)")
        lx = max(5, min(w - 120, ix + 12))
        ly = max(15, min(h - 5, iy + 12))
        cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2)

    cv2.imwrite(str(save_path), img)
    return {"bbox": (x1, y1, x2, y2),
            "keypoints": keypoints, "class_id": class_id}


def validate_split_with_opencv(split_dir: Path, limit: int,
                               out_dir: Path) -> Dict:
    """对某个 split（train/valid/test）执行传统 OpenCV 可视化校验。

    校验内容：
        - 标签文件是否与图片一一对应；
        - 标签字段数量、数值范围（归一化坐标 0~1）是否合法；
        - 反归一化后的 bbox / 关键点是否落在图像范围内；
        - 生成带标注的可视化图片供人工目检关键点顺序。
    """
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.exists():
        return {"error": f"缺少 images 目录: {images_dir}"}
    if not labels_dir.exists():
        return {"error": f"缺少 labels 目录: {labels_dir}"}

    images = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if limit and limit > 0:
        images = images[:limit]

    save_dir = out_dir / split_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0, "ok": 0, "label_missing": 0, "parse_error": 0,
        "bbox_out_of_range": 0, "kpt_out_of_range": 0,
        "saved": 0,
    }
    saved_paths: List[Path] = []

    for img_path in images:
        stats["total"] += 1
        label_path = labels_dir / f"{img_path.stem}.txt"
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[validate] 无法读取图片: {img_path.name}")
            continue
        if not label_path.exists():
            stats["label_missing"] += 1
            print(f"[validate] 缺少标签: {label_path.name}")
            continue

        h, w = image.shape[:2]
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            for line in lines:
                _, bbox, keypoints = parse_label_line(line, w, h)
                x1, y1, x2, y2 = bbox
                if not (-5 <= x1 <= w + 5 and -5 <= y1 <= h + 5 and
                        -5 <= x2 <= w + 5 and -5 <= y2 <= h + 5):
                    stats["bbox_out_of_range"] += 1
                for (px, py), vis in keypoints:
                    if not (-5 <= px <= w + 5 and -5 <= py <= h + 5):
                        stats["kpt_out_of_range"] += 1
                draw_validation_image(image, line, save_dir / img_path.name)
                stats["saved"] += 1
            stats["ok"] += 1
            saved_paths.append(save_dir / img_path.name)
        except Exception as exc:
            stats["parse_error"] += 1
            print(f"[validate] 标签解析失败 {label_path.name}: {exc}")

    print(f"[validate] {split_dir.name} 校验完成: {stats}")
    return {"stats": stats, "saved_paths": saved_paths}


def build_montage(image_paths: List[Path], out_path: Path,
                  cols: int = 4, rows: int = 4, cell: int = 320) -> None:
    """把校验图拼成网格大图，方便一次性目检多张样本。"""
    if not image_paths:
        return
    canvas = np.empty((cell * rows, cell * cols, 3), dtype=np.uint8)
    canvas.fill(114)
    for i, p in enumerate(image_paths[:cols * rows]):
        img = cv2.imread(str(p))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        scale = min(cell / ih, cell / iw)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        ox = c * cell + (cell - nw) // 2
        oy = r * cell + (cell - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"[validate] 已生成蒙太奇图: {out_path}")


# ---------------------------------------------------------------------------
# 3. 训练与导出
# ---------------------------------------------------------------------------
def train_pose(data_yaml: Path, args: argparse.Namespace) -> Path:
    """训练 YOLOv8-Pose 关键点模型，返回 best.pt 路径。"""
    from ultralytics import YOLO

    print(f"[train] 加载预训练模型: {args.model}")
    model = YOLO(args.model)
    project = PROJECT_DIR / "runs" / "pose"
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(project),
        name=args.name,
        exist_ok=args.exist_ok,
        patience=args.patience,
        workers=args.workers,
        fliplr=args.fliplr,   # 混合表盘几何数据集默认关闭水平翻转增强
        cache=args.cache,     # 图像缓存策略：off / ram / disk
        lr0=args.lr0,         # 初始学习率（续训/微调建议 0.001~0.003）
        cos_lr=args.cos_lr,   # 余弦学习率调度（默认 False=线性衰减）
        deterministic=True,
        plots=False,          # 关闭批次预览图绘制，规避 OpenCV 内存分配失败（Insufficient memory）
    )
    best = project / args.name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"训练完成但未找到权重: {best}")

    # 复制一份到 models/，与 config/gauge.yaml 的默认路径保持一致
    DEFAULT_MODELS_DIR.mkdir(exist_ok=True)
    target = DEFAULT_MODELS_DIR / "gauge_pose.pt"
    shutil.copy2(str(best), str(target))
    print(f"[train] 已复制权重到: {target}")
    if args.model_suffix:
        suffixed = DEFAULT_MODELS_DIR / f"gauge_pose_{args.model_suffix}.pt"
        shutil.copy2(str(best), str(suffixed))
        print(f"[train] 已复制权重到: {suffixed}")
    return best


def export_onnx(best_pt: Path, imgsz: int, simplify: bool = True,
                suffix: str = "") -> Path:
    """把 best.pt 导出为 ONNX 并复制到 models/（可选带后缀）。"""
    from ultralytics import YOLO

    model = YOLO(str(best_pt))
    try:
        model.export(format="onnx", imgsz=imgsz, opset=12,
                     simplify=simplify, dynamic=False)
    except Exception as exc:
        print(f"[train] simplify 导出失败，回退为不简化: {exc}")
        model.export(format="onnx", imgsz=imgsz, opset=12,
                     simplify=False, dynamic=False)
    src = best_pt.with_suffix(".onnx")
    target = DEFAULT_MODELS_DIR / "gauge_pose.onnx"
    if src.exists():
        shutil.copy2(str(src), str(target))
        print(f"[train] 已导出 ONNX: {target}")
        if suffix:
            suffixed = DEFAULT_MODELS_DIR / f"gauge_pose_{suffix}.onnx"
            shutil.copy2(str(src), str(suffixed))
            print(f"[train] 已导出 ONNX: {suffixed}")
    return target


def export_tflite(best_pt: Path, imgsz: int) -> Optional[Path]:
    """把 best.pt 导出为 TFLite 并复制到 models/gauge_pose.tflite。

    注意：TFLite 导出需要 tensorflow，边缘设备可跳过此步（仅推理 TFLite）。
    """
    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"[train] 跳过 TFLite 导出（ultralytics 不可用）: {exc}")
        return None
    try:
        model = YOLO(str(best_pt))
        result = model.export(format="tflite", imgsz=imgsz)
    except Exception as exc:
        print(f"[train] 跳过 TFLite 导出（需要 tensorflow）: {exc}")
        return None

    candidates = list(result) if isinstance(result, (list, tuple)) else [result]
    src = next((Path(c) for c in candidates if Path(c).exists()), None)
    if src is None:  # 兼容部分版本的导出路径
        matches = sorted((best_pt.parent.parent).rglob("*.tflite"))
        src = matches[0] if matches else None
    if src is None:
        print("[train] 未找到 TFLite 导出产物")
        return None
    target = DEFAULT_MODELS_DIR / "gauge_pose.tflite"
    shutil.copy2(str(src), str(target))
    print(f"[train] 已导出 TFLite: {target}")
    return target


def resume_pose(args: argparse.Namespace) -> Path:
    """从上次中断的 runs/pose/<name>/weights/last.pt 续训，返回 best.pt 路径。

    续训会沿用训练目录 args.yaml 里保存的全部参数（epochs / lr0 / cos_lr /
    batch / imgsz 等），从断点继续跑到原定的总 epoch 数，无需重新指定。
    """
    from ultralytics import YOLO

    project = PROJECT_DIR / "runs" / "pose"
    ckpt = project / args.name / "weights" / "last.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"未找到可续训的检查点: {ckpt}\n请确认 --name 与上次训练一致，"
            "且至少完成了一个 epoch（存在 last.pt）")
    print(f"[train] 从检查点续训: {ckpt}")
    YOLO(str(ckpt)).train(resume=True)

    best = project / args.name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"续训完成但未找到权重: {best}")

    # 复制一份到 models/（与 train_pose 保持一致，可选带后缀）
    DEFAULT_MODELS_DIR.mkdir(exist_ok=True)
    target = DEFAULT_MODELS_DIR / "gauge_pose.pt"
    shutil.copy2(str(best), str(target))
    print(f"[train] 已复制权重到: {target}")
    if args.model_suffix:
        suffixed = DEFAULT_MODELS_DIR / f"gauge_pose_{args.model_suffix}.pt"
        shutil.copy2(str(best), str(suffixed))
        print(f"[train] 已复制权重到: {suffixed}")
    return best


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gauge 数据集校验 + YOLOv8-Pose 训练 + 导出")
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML),
                        help="数据集 data.yaml 路径")
    parser.add_argument("--model", default="yolov8n-pose.pt",
                        help="预训练权重或模型结构（yolov8n-pose.pt / yolo11n-pose.pt）")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=8,
                        help="批大小；-1=自动探测显存上限（8GB 显存卡推荐），"
                             "也可用 --batch 8 --imgsz 512 降低显存占用")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="0=第一块 GPU，cpu=CPU")
    parser.add_argument("--name", default="gauge", help="训练输出目录名")
    parser.add_argument("--exist-ok", action="store_true",
                        help="允许覆盖同名训练输出目录")
    parser.add_argument("--patience", type=int, default=60,
                        help="早停耐心值")
    parser.add_argument("--workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--cache", default="off", choices=["off", "ram", "disk"],
                        help="图像缓存策略：off=不缓存（默认，省内存）；"
                             "ram=缓存到内存（大内存机器）；disk=缓存到磁盘（需大量磁盘空间）")
    parser.add_argument("--fliplr", type=float, default=0.0,
                        help="水平翻转增强概率（混合表盘几何数据集默认 0）")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="初始学习率（续训/微调建议 0.001~0.003）")
    parser.add_argument("--cos-lr", action="store_true",
                        help="使用余弦学习率调度（默认 False=线性衰减）")

    # 校验相关
    parser.add_argument("--validate-only", action="store_true",
                        help="只做 OpenCV 数据可视化校验，不训练")
    parser.add_argument("--skip-validate", action="store_true",
                        help="跳过训练前的 OpenCV 校验")
    parser.add_argument("--limit", type=int, default=20,
                        help="每个 split 校验的样本数（0=全部）")
    parser.add_argument("--out", default=str(PROJECT_DIR / "runs" / "validation"),
                        help="校验图输出目录")

    # 导出相关
    parser.add_argument("--export-onnx", action="store_true",
                        help="训练结束后导出 ONNX 到 models/")
    parser.add_argument("--export-tflite", action="store_true",
                        help="训练结束后导出 TFLite 到 models/（需 tensorflow）")
    parser.add_argument("--no-simplify", action="store_false", dest="simplify",
                        default=True,
                        help="禁用 ONNX simplify 导出（默认开启，需 onnxslim）")
    parser.add_argument("--model-suffix", default="",
                        help="复制到 models/ 时的文件名后缀，如 e120 -> "
                             "gauge_pose_e120.pt / gauge_pose_e120.onnx")
    parser.add_argument("--resume", action="store_true",
                        help="从 runs/pose/<name>/weights/last.pt 续训"
                             "（沿用上次全部参数，自动跳过数据校验）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = fix_data_yaml(Path(args.data))
    out_dir = Path(args.out)

    # ---- 传统 OpenCV 数据校验（续训时自动跳过） ----
    if not args.skip_validate and not args.resume:
        dataset_dir = data_yaml.parent
        montage_paths = []
        for split in ("train", "valid", "test"):
            split_dir = dataset_dir / split
            if not split_dir.exists():
                print(f"[validate] 跳过不存在的 split: {split_dir}")
                continue
            result = validate_split_with_opencv(split_dir, args.limit, out_dir)
            if "saved_paths" in result and result["saved_paths"]:
                montage_paths.extend(result["saved_paths"][:16])
        if montage_paths:
            build_montage(montage_paths, out_dir / "validation_montage.jpg")
        print("[validate] 完成。请人工目检校验图，确认 4 个关键点顺序"
              "（min=0/max=1/tip=2/center=3）与检测框位置正确后再训练。")
        if args.validate_only:
            return

    # ---- 训练 / 续训 ----
    if args.resume:
        best = resume_pose(args)
    else:
        best = train_pose(data_yaml, args)

    # ---- 导出 ----
    if args.export_onnx:
        export_onnx(best, args.imgsz, simplify=args.simplify,
                    suffix=args.model_suffix)
    if args.export_tflite:
        export_tflite(best, args.imgsz)

    print("[train] 全部完成。推理前请把 config/gauge.yaml 的 model.backend "
          "切换为 ultralytics / onnx / tflite，并确认模型路径。")
    if args.model_suffix:
        print(f"[train] 提示：新模型文件位于 models/gauge_pose_{args.model_suffix}.pt"
              f" 与 models/gauge_pose_{args.model_suffix}.onnx，"
              "如需使用请把 config/gauge.yaml 的 model_path 指过去。")


if __name__ == "__main__":
    main()
