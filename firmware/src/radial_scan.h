#pragma once

#include <cstdint>

namespace gauge {

struct ScanParams {
    float inner_ratio = 0.30f;   // 扫描带内半径 / 表盘半径
    float outer_ratio = 0.90f;   // 扫描带外半径 / 表盘半径
    float step_deg = 0.5f;       // 角度步长（度）
    float min_score = 0.35f;     // 有效得分阈值
    float max_flat_ratio = 0.15f;// 平坦度上限
};

struct ScanResult {
    bool ok = false;
    float angle_deg = 0.0f;      // 指针方向 [0,360)，图像坐标（y 向下）
    float confidence = 0.0f;     // 有效得分 peak*(1-flat)
};

// 径向扫描检测指针方向（与 Python src/refiner.py 同算法）
// gray: w*h 灰度图；cx/cy/radius 为表盘中心与半径（像素）
ScanResult scan_needle_angle(const uint8_t* gray, int w, int h,
                             float cx, float cy, float radius,
                             const ScanParams& p = ScanParams());

}  // namespace gauge
