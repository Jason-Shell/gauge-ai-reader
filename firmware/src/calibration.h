#pragma once

struct Calibration {
    bool valid = false;
    float cx = 0.0f;        // 表盘中心 x
    float cy = 0.0f;        // 表盘中心 y
    float radius = 0.0f;    // 表盘半径（像素）
    float min_angle = 0.0f; // 刻度起点端角度 [0,360)
    float max_angle = 0.0f; // 刻度终点端角度 [0,360)
    int sweep = 0;          // 0=顺时针 1=逆时针
    float bar_min = 0.0f;
    float bar_max = 250.0f;
    float psi_min = 0.0f;
    float psi_max = 3625.9f;
};

const Calibration& get_calibration();
void set_calibration(const Calibration& c);

// 从 NVS 加载 / 保存 / 清除标定，成功返回 true
bool load_calibration();
void save_calibration();
void clear_calibration();
