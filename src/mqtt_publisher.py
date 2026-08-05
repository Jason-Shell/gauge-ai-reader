# -*- coding: utf-8 -*-
"""mqtt_publisher.py —— 可选 MQTT 读数上传（默认关闭，缺 paho-mqtt 时自动禁用）。

设计要点：
    - 纯增量功能：不安装 paho-mqtt 或 broker 不可达时，程序照常运行，仅打印提示；
    - 流式场景按时间节流（interval_sec），避免每帧都发消息；
    - 使用 loop_start() 异步发送，不阻塞推理主循环；
    - 负载为 JSON：{source, ts, readings: [{index, bar, psi, ratio, conf, keypoints}]}。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from reader import Reading


class MqttPublisher:
    """封装 paho-mqtt 客户端，对外只暴露 maybe_publish() 与 close()。"""

    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self._client = None
        self._last_ts = 0.0
        self.topic = str(cfg.get("topic", "gauge/reading"))
        self.qos = int(cfg.get("qos", 0))
        self.interval = max(0.0, float(cfg.get("interval_sec", 1.0)))
        if not self.enabled:
            return

        host = str(cfg.get("host", "localhost"))
        port = int(cfg.get("port", 1883))
        client_id = str(cfg.get("client_id", "gauge-reader")) or None
        try:
            import paho.mqtt.client as mqtt

            self._client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
            if cfg.get("username"):
                self._client.username_pw_set(
                    str(cfg["username"]), str(cfg.get("password", "")))
            self._client.connect(host, port, keepalive=30)
            self._client.loop_start()
            print(f"[mqtt] 已连接 {host}:{port} -> topic: {self.topic}")
        except Exception as exc:  # 缺依赖 / broker 不可达均静默降级
            print(f"[mqtt] 不可用，已禁用上传: {exc}")
            self._client = None

    @property
    def active(self) -> bool:
        """是否真正连接成功并允许发布。"""
        return self._client is not None

    @staticmethod
    def _payload(source: str, readings: List[Reading]) -> str:
        items = []
        for i, r in enumerate(readings):
            items.append({
                "index": i,
                "bar": r.primary_value,
                "psi": r.secondary_value,
                "ratio": r.ratio,
                "conf": r.conf,
                "error": r.error,
                "refine_used": r.refine_used,
                "refine_conf": r.refine_conf,
                "keypoints": [[round(x, 2), round(y, 2)] for x, y in r.keypoints],
            })
        payload = {
            "source": str(source),
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "readings": items,
        }
        return json.dumps(payload, ensure_ascii=False)

    def maybe_publish(self, source, readings: List[Reading]) -> None:
        """按时间节流发布：图片模式发一次，视频/摄像头模式按 interval_sec 限频。"""
        if not self.active:
            return
        now = time.monotonic()
        if now - self._last_ts < self.interval:
            return
        self._last_ts = now
        try:
            self._client.publish(self.topic, self._payload(source, readings),
                                 qos=self.qos)
        except Exception as exc:
            print(f"[mqtt] 发送失败: {exc}")

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._client = None
