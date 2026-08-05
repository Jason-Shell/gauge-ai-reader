#include "calibration.h"

#include <cstring>

#include <Preferences.h>

static Calibration s_cal;

const Calibration& get_calibration() { return s_cal; }
void set_calibration(const Calibration& c) { s_cal = c; }

bool load_calibration() {
    Preferences prefs;
    if (!prefs.begin("gauge_cal", true)) return false;
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
