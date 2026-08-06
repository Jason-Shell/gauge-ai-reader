# -*- coding: utf-8 -*-
"""capture_images.py —— 通过 WiFi 从 ESP32-S3 固件批量抓取表盘图像。

配合固件 Web 接口 `GET /capture`（返回 JPEG）使用，用于“先传图、后标注”：
把实拍表盘照片按时间戳保存到本地目录，之后再标定读数
（eval_reading.py 的 CSV: path,bar[,psi]，或人工标注，作为路线 B 训练数据）。

用法（项目根目录，用 .python312）：
    & "D:\\JasonXie\\Code-OpenCV\\Project\\.python312\\python.exe" scripts\\capture_images.py ^
        --url http://192.168.101.126 --out datasets/gauge_real --count 30 --interval 1

参数：
    --url      设备地址（IP 或 mDNS，如 http://gauge.local）
    --out      保存目录（默认 datasets/gauge_real，不入库）
    --count    抓取张数（默认 1）
    --interval 相邻两次抓取间隔秒（默认 1，0=不等待）
    --prefix   文件名前缀（默认 gauge）
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.request import urlopen


def grab(url: str, out_dir: Path, count: int, interval: float,
         prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for i in range(count):
        try:
            with urlopen(url.rstrip("/") + "/capture", timeout=10) as resp:
                data = resp.read()
        except Exception as exc:
            print(f"[{i + 1}/{count}] 抓取失败: {exc}")
            continue
        if len(data) < 4 or data[:2] != b"\xff\xd8":
            print(f"[{i + 1}/{count}] 响应不是 JPEG（{len(data)} 字节）")
            continue
        path = out_dir / (
            f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{i:03d}.jpg")
        path.write_bytes(data)
        ok += 1
        print(f"[{i + 1}/{count}] 已保存 {path}（{len(data) / 1024:.0f} KB）")
        if i < count - 1 and interval > 0:
            time.sleep(interval)
    print(f"完成：成功 {ok}/{count} 张 -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 ESP32-S3 固件批量抓取表盘图像（GET /capture）")
    parser.add_argument("--url", default="http://gauge.local",
                        help="设备地址，默认 http://gauge.local")
    parser.add_argument("--out", default="datasets/gauge_real",
                        help="保存目录（默认 datasets/gauge_real）")
    parser.add_argument("--count", type=int, default=1,
                        help="抓取张数（默认 1）")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="抓取间隔秒（默认 1，0=不等待）")
    parser.add_argument("--prefix", default="gauge",
                        help="文件名前缀（默认 gauge）")
    args = parser.parse_args()
    grab(args.url, Path(args.out), args.count, args.interval, args.prefix)


if __name__ == "__main__":
    main()
