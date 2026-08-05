#include "radial_scan.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace gauge {
namespace {

constexpr float kPi = 3.14159265358979323846f;

// 沿射线统计“从内缘开始”且满足谓词的连续段长度占比
// want_dark=true 时数暗段（gray < thr），false 时数亮段（gray >= thr）
inline float run_from_start(const uint8_t* gray, int w, int h,
                            float cx, float cy, float r_in, float r_out,
                            float theta, uint8_t thr, bool want_dark, int n) {
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);
    int count = 0;
    for (int i = 0; i < n; ++i) {
        float r = r_in + (r_out - r_in) * (float)i / (float)(n - 1);
        int x = (int)lrintf(cx + r * cos_t);
        int y = (int)lrintf(cy + r * sin_t);
        if (x < 0) x = 0;
        if (x >= w) x = w - 1;
        if (y < 0) y = 0;
        if (y >= h) y = h - 1;
        if ((gray[y * w + x] < thr) == want_dark) {
            ++count;
        } else {
            break;
        }
    }
    return (float)count / (float)n;
}

// 抛物线插值求亚度峰值
inline float interp_peak(const float* scores, int n_angles, float step_deg) {
    int i1 = 0;
    for (int i = 1; i < n_angles; ++i) {
        if (scores[i] > scores[i1]) i1 = i;
    }
    int i0 = (i1 - 1 + n_angles) % n_angles;
    int i2 = (i1 + 1) % n_angles;
    float s0 = scores[i0], s1 = scores[i1], s2 = scores[i2];
    float denom = s0 - 2.0f * s1 + s2;
    float delta = (fabsf(denom) > 1e-9f) ? 0.5f * (s0 - s2) / denom : 0.0f;
    float angle = ((float)i1 + delta) * step_deg;
    return fmodf(angle + 3600.0f, 360.0f);
}

}  // namespace

ScanResult scan_needle_angle(const uint8_t* gray, int w, int h,
                             float cx, float cy, float radius,
                             const ScanParams& p) {
    ScanResult result;
    float r_in = radius * p.inner_ratio;
    float r_out = radius * p.outer_ratio;
    if (r_out - r_in < 5.0f || radius < 8.0f || w < 8 || h < 8) {
        return result;
    }

    // ---- 环带内像素直方图 -> Otsu 阈值（含 5/95 分位钳制）----
    int hist[256] = {0};
    float r2_in = r_in * r_in;
    float r2_out = r_out * r_out;
    int total = 0;
    for (int y = 0; y < h; ++y) {
        float dy = (float)y - cy;
        for (int x = 0; x < w; ++x) {
            float dx = (float)x - cx;
            float d2 = dx * dx + dy * dy;
            if (d2 >= r2_in && d2 <= r2_out) {
                ++hist[gray[y * w + x]];
                ++total;
            }
        }
    }
    if (total < 50) return result;

    // Otsu
    float sum = 0.0f;
    for (int t = 0; t < 256; ++t) sum += (float)t * hist[t];
    float sum_b = 0.0f;
    int w_b = 0;
    float best_var = -1.0f;
    int thr = 0;
    for (int t = 0; t < 256; ++t) {
        w_b += hist[t];
        if (w_b == 0) continue;
        int w_f = total - w_b;
        if (w_f == 0) break;
        sum_b += (float)t * hist[t];
        float m_b = sum_b / w_b;
        float m_f = (sum - sum_b) / w_f;
        float var = (float)w_b * (float)w_f * (m_b - m_f) * (m_b - m_f);
        if (var > best_var) {
            best_var = var;
            thr = t;
        }
    }
    // 5/95 分位钳制，防纯 0/255 二值图退化
    int p5 = 0, p95 = 255;
    {
        int cum = 0;
        bool got_p5 = false, got_p95 = false;
        for (int t = 0; t < 256; ++t) {
            cum += hist[t];
            if (!got_p5 && cum * 20 >= total) { p5 = t; got_p5 = true; }
            if (!got_p95 && cum * 20 >= total * 19) { p95 = t; got_p95 = true; break; }
        }
    }
    if (thr < p5) thr = p5;
    if (thr > p95) thr = p95;
    uint8_t thr_u = (uint8_t)thr;

    // ---- 双极性径向扫描 ----
    int n_angles = (int)lrintf(360.0f / p.step_deg);
    int n_pts = std::max(8, (int)(r_out - r_in));
    float* scores = (float*)malloc((size_t)n_angles * sizeof(float));
    if (!scores) return result;

    float best_eff = -1.0f;
    float best_angle = 0.0f;
    float best_peak = 0.0f;

    for (int polarity = 0; polarity < 2; ++polarity) {
        bool want_dark = (polarity == 0);  // 0：暗为“针”；1：亮为“针”
        for (int i = 0; i < n_angles; ++i) {
            float theta = (float)i * p.step_deg * kPi / 180.0f;
            float v = run_from_start(gray, w, h, cx, cy, r_in, r_out,
                                     theta, thr_u, want_dark, n_pts);
            scores[i] = v;
        }
        float peak = 0.0f;
        for (int i = 0; i < n_angles; ++i) {
            if (scores[i] > peak) peak = scores[i];
        }
        int above = 0;
        for (int i = 0; i < n_angles; ++i) {
            if (scores[i] >= 0.8f * peak) ++above;
        }
        float flat = (float)above / (float)n_angles;
        float eff = peak * (1.0f - flat);
        if (eff > best_eff) {
            best_eff = eff;
            best_peak = peak;
            best_angle = interp_peak(scores, n_angles, p.step_deg);
        }
    }
    free(scores);

    if (best_eff < p.min_score || best_peak < p.min_score) return result;
    result.ok = true;
    result.angle_deg = best_angle;
    result.confidence = best_eff;
    return result;
}

}  // namespace gauge
