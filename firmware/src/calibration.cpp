#include "calibration.h"

#include <cstring>

#include <Preferences.h>

static Calibration s_cal;

const Calibration& get_calibration() { return s_cal; }
void set_calibration(const Calibration& c) { s_cal = c; }

bool load_calibration() {
    Preferences prefs;
    // 用读写模式打开：首次上电时命名空间不存在，只读打开会触发
    // Preferences 库的 nvs_open NOT_FOUND 错误日志（无害但误导）。
    // 读写打开会自动创建命名空间，再用“无 cal 键”判断未标定。
    if (!prefs.begin("gauge_cal", false)) return false;
    size_t len = prefs.getBytesLength("cal");
    if (len != sizeof(Calibration)) {
        prefs.end();
        return false;
    }
    Calibration tmp;
    prefs.getBytes("cal", &tmp, sizeof(tmp));
    prefs.end();
    if (!tmp.valid || tmp.radius < 8.0f) return false;
    s_cal = tmp;
    return true;
}

void save_calibration() {
    Preferences prefs;
    if (!prefs.begin("gauge_cal", false)) return;
    s_cal.valid = true;
    prefs.putBytes("cal", &s_cal, sizeof(s_cal));
    prefs.end();
}

void clear_calibration() {
    Preferences prefs;
    if (prefs.begin("gauge_cal", false)) {
        prefs.remove("cal");
        prefs.end();
    }
    s_cal = Calibration();
}
