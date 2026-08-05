#pragma once

#include "esp_camera.h"

// ==================== 固件级配置（改完重新编译烧录） ====================

// WiFi：STA 模式，两种配网方式（优先级从上到下）：
//   1. 本地写死：在 firmware/src/ 下创建 wifi_credentials.h（复制
//      wifi_credentials.example.h 并填入真实 SSID/密码）。该文件已被
//      .gitignore 排除，不会进入仓库，密码不会公开；
//   2. 串口配网：未创建上述文件时，用 WIFI SET <ssid> <pass> 写入 NVS。
#if __has_include("wifi_credentials.h")
#include "wifi_credentials.h"
#else
#define STA_SSID ""
#define STA_PASS ""
#endif

#define STA_HOST          "gauge"       // mDNS 主机名：http://gauge.local/
#define WIFI_TIMEOUT_MS   15000         // 连接超时（毫秒），超时后仅串口可用

// 摄像头：RGB565 QVGA（320x240），无需 JPEG 解码即可做灰度扫描
#define CAM_FRAME     FRAMESIZE_QVGA
#define CAM_FORMAT    PIXFORMAT_RGB565

// 自动读数间隔（毫秒）
#define READ_INTERVAL_MS  500

// 未校准时使用的默认表盘参数（用 CAL CENTER AUTO 或手动标定覆盖）
#define DEFAULT_RADIUS_RATIO 0.45f   // 半径 = min(w,h) * 该比例

// 双量程（与 config/gauge.yaml 保持一致）
#define GAUGE_BAR_MIN  0.0f
#define GAUGE_BAR_MAX  250.0f
#define GAUGE_PSI_MIN  0.0f
#define GAUGE_PSI_MAX  3625.9f

// 默认扫掠方向：0=顺时针 1=逆时针（用串口 CAL SWEEP 修改）
#define DEFAULT_SWEEP  0
