#pragma once

#include <cstdint>

// 初始化 OV2640（针脚见 camera_utils.cpp，与旧工程一致）
bool init_camera();

// 画质/传感器参数优化
void optimize_camera_image();

// 采集一帧 RGB565 并转换为灰度图
// 返回内部缓冲指针（下一次调用会覆盖），失败返回 nullptr；
// 宽高写入 out_w / out_h
const uint8_t* capture_grayscale(int* out_w, int* out_h);
