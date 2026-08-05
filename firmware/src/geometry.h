#pragma once

#include <cmath>

namespace gauge {

// 与 Python src/geometry.py 一一对应的纯数学实现（无 Arduino 依赖，可主机测试）

inline float normalize_angle(float a) {
    a = fmodf(a, 360.0f);
    if (a < 0.0f) a += 360.0f;
    return a;
}

inline float relative_angle(float from_deg, float to_deg) {
    return fmodf(to_deg - from_deg + 360.0f, 360.0f);
}

// sweep: 0=clockwise，1=counterclockwise，2=auto
// 返回比例 [0,1]；量程跨度过小 / 参数无效时返回 -1
inline float calculate_ratio(float min_angle, float max_angle,
                             float tip_angle, int sweep) {
    const float MIN_SPAN_DEG = 5.0f;
    const float EPS = 1e-6f;
    if (sweep < 0 || sweep > 2) return -1.0f;
    float a_min = normalize_angle(min_angle);
    float a_max = normalize_angle(max_angle);
    float a_tip = normalize_angle(tip_angle);

    float span_cw = relative_angle(a_min, a_max);
    float span_ccw = relative_angle(a_max, a_min);
    float dist_cw = relative_angle(a_min, a_tip);
    float dist_ccw = relative_angle(a_tip, a_min);

    float ratio = -1.0f;
    if (sweep == 0) {          // clockwise
        if (span_cw < MIN_SPAN_DEG) return -1.0f;
        ratio = dist_cw / span_cw;
    } else if (sweep == 1) {   // counterclockwise
        if (span_ccw < MIN_SPAN_DEG) return -1.0f;
        ratio = dist_ccw / span_ccw;
    } else {                   // sweep==2 auto：选择指针所在弧；同时落在两弧时取短弧
        bool in_cw = dist_cw <= span_cw + EPS;
        bool in_ccw = dist_ccw <= span_ccw + EPS;
        if (in_cw && !in_ccw) {
            if (span_cw < MIN_SPAN_DEG) return -1.0f;
            ratio = dist_cw / span_cw;
        } else if (in_ccw && !in_cw) {
            if (span_ccw < MIN_SPAN_DEG) return -1.0f;
            ratio = dist_ccw / span_ccw;
        } else if (in_cw && in_ccw) {
            if (span_cw <= span_ccw) {
                ratio = dist_cw / span_cw;
            } else {
                ratio = dist_ccw / span_ccw;
            }
        } else {
            return -1.0f;
        }
    }
    if (ratio < 0.0f) ratio = 0.0f;
    if (ratio > 1.0f) ratio = 1.0f;
    return ratio;
}

inline float calculate_value(float min_v, float max_v, float ratio) {
    return min_v + ratio * (max_v - min_v);
}

inline float round_value(float value, float step) {
    if (step <= 0.0f) return value;
    return roundf(value / step) * step;
}

}  // namespace gauge
