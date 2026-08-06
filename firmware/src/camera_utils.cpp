#include "Arduino.h"

#include "camera_utils.h"

#include <cstring>

#include "esp_camera.h"
#include "esp_heap_caps.h"

// ==================== OV2640 引脚（与旧工程 my_first_esp32s3 一致） ====================
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  15
#define SIOD_GPIO_NUM  4
#define SIOC_GPIO_NUM  5
#define Y2_GPIO_NUM    11
#define Y3_GPIO_NUM    9
#define Y4_GPIO_NUM    8
#define Y5_GPIO_NUM    10
#define Y6_GPIO_NUM    12
#define Y7_GPIO_NUM    18
#define Y8_GPIO_NUM    17
#define Y9_GPIO_NUM    16
#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM  7
#define PCLK_GPIO_NUM  13

static uint8_t* s_gray = nullptr;  // 灰度缓冲（PSRAM）

camera_config_t make_camera_config() {
    const bool has_psram = psramFound();
    camera_config_t config = {};
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.ledc_timer = LEDC_TIMER_0;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 10;
    config.fb_count = 1;
    config.fb_location = has_psram ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
    config.grab_mode = has_psram ? CAMERA_GRAB_LATEST : CAMERA_GRAB_WHEN_EMPTY;
    return config;
}

bool init_camera() {
    Serial.println("正在初始化摄像头...");
    Serial.printf("Flash: %u B  PSRAM: %u B  free heap: %u\n",
                  ESP.getFlashChipSize(), ESP.getPsramSize(), ESP.getFreeHeap());
    camera_config_t camera_config = make_camera_config();
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        Serial.printf("摄像头初始化失败：0x%x\n", err);
        return false;
    }
    sensor_t* sensor = esp_camera_sensor_get();
    if (sensor) {
        Serial.printf("Camera PID: 0x%02X\n", sensor->id.PID);
        if (sensor->id.PID != OV2640_PID) {
            Serial.println("警告：检测到的传感器不是 OV2640");
        }
    }
    Serial.println("摄像头初始化成功！");
    return true;
}

void optimize_camera_image() {
    sensor_t* s = esp_camera_sensor_get();
    if (!s) {
        Serial.println("警告：无法获取传感器，跳过画质优化");
        return;
    }
    s->set_framesize(s, FRAMESIZE_VGA);
    s->set_quality(s, 10);
    s->set_brightness(s, 0);
    s->set_contrast(s, 2);
    s->set_saturation(s, 0);
    s->set_sharpness(s, 2);
    s->set_denoise(s, 1);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_wb_mode(s, 0);
    s->set_exposure_ctrl(s, 1);
    s->set_gain_ctrl(s, 1);
    s->set_aec2(s, 1);
    s->set_gainceiling(s, GAINCEILING_8X);
    s->set_bpc(s, 1);
    s->set_wpc(s, 1);
    s->set_raw_gma(s, 1);
    s->set_lenc(s, 1);
    s->set_dcw(s, 1);
    s->set_vflip(s, 0);
    s->set_hmirror(s, 0);
    Serial.println("画质优化完成（VGA RGB565）");
}

const uint8_t* capture_grayscale(int* out_w, int* out_h) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) return nullptr;
    const int w = fb->width;
    const int h = fb->height;
    if (!s_gray) {
        s_gray = (uint8_t*)heap_caps_malloc((size_t)w * h, MALLOC_CAP_SPIRAM);
        if (!s_gray) {
            esp_camera_fb_return(fb);
            return nullptr;
        }
    }
    if (fb->format == PIXFORMAT_RGB565) {
        const uint16_t* px = (const uint16_t*)fb->buf;
        for (int i = 0; i < w * h; ++i) {
            uint16_t p = px[i];
            uint8_t r = (p >> 11) & 0x1F;
            uint8_t g = (p >> 5) & 0x3F;
            uint8_t b = p & 0x1F;
            uint8_t r8 = (uint8_t)((r << 3) | (r >> 2));
            uint8_t g8 = (uint8_t)((g << 2) | (g >> 4));
            uint8_t b8 = (uint8_t)((b << 3) | (b >> 2));
            s_gray[i] = (uint8_t)((r8 * 77u + g8 * 150u + b8 * 29u) >> 8);
        }
    } else if (fb->format == PIXFORMAT_GRAYSCALE) {
        memcpy(s_gray, fb->buf, (size_t)w * h);
    } else {
        esp_camera_fb_return(fb);
        return nullptr;
    }
    esp_camera_fb_return(fb);
    if (out_w) *out_w = w;
    if (out_h) *out_h = h;
    return s_gray;
}
