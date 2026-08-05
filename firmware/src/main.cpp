#include "Arduino.h"
#include "esp_camera.h"
#include "img_converters.h"
#include "WiFi.h"
#include "ESPmDNS.h"
#include "WebServer.h"
#include "Preferences.h"

#include "config.h"
#include "geometry.h"
#include "radial_scan.h"
#include "calibration.h"
#include "camera_utils.h"

// ==================== 全局状态 ====================
static WebServer server(80);

struct Reading {
    bool ok = false;
    float bar = 0.0f, psi = 0.0f, ratio = 0.0f;
    float angle = 0.0f, conf = 0.0f;
    unsigned long ts_ms = 0;
    String error;
};
static Reading s_reading;
static unsigned long s_last_read_ms = 0;
static String s_serial_buf;

// ==================== WiFi（STA 模式，串口配网 + NVS） ====================
static String s_wifi_ssid, s_wifi_pass;
static bool s_wifi_diag_printed = false;

// 连接失败时打印一次断开原因码（201=找不到 SSID，202=认证失败，200=信标超时）
static void on_wifi_event(arduino_event_id_t event, arduino_event_info_t info) {
    if (event != ARDUINO_EVENT_WIFI_STA_DISCONNECTED) return;
    if (s_wifi_diag_printed) return;   // 每次开机只提示一次，避免重连刷屏
    s_wifi_diag_printed = true;
    uint8_t reason = (uint8_t)info.wifi_sta_disconnected.reason;
    Serial.printf("[wifi] 连接失败，断开原因码 %u\n", reason);
    switch (reason) {
        case 201:
            Serial.println("[wifi] 未找到该 SSID：可能仅 5GHz、距离太远或 SSID 隐藏");
            break;
        case 202:
            Serial.println("[wifi] 认证失败：密码错误，或安全模式不兼容");
            break;
        case 15:
        case 204:
            Serial.println("[wifi] 4-way 握手超时：多为密码错误");
            break;
        case 200:
            Serial.println("[wifi] BEACON_TIMEOUT：信号弱或丢失");
            break;
        default:
            break;
    }
}

static bool load_wifi_config() {
    Preferences prefs;
    // 读写模式自动创建命名空间，避免首启只读打开报 nvs_open NOT_FOUND
    if (!prefs.begin("gauge_wifi", false)) return false;
    s_wifi_ssid = prefs.isKey("ssid") ? prefs.getString("ssid", "") : "";
    s_wifi_pass = prefs.isKey("pass") ? prefs.getString("pass", "") : "";
    prefs.end();
    return s_wifi_ssid.length() > 0;
}

static void save_wifi_config(const String& ssid, const String& pass) {
    Preferences prefs;
    if (!prefs.begin("gauge_wifi", false)) return;
    prefs.putString("ssid", ssid);
    prefs.putString("pass", pass);
    prefs.end();
    s_wifi_ssid = ssid;
    s_wifi_pass = pass;
}

static void clear_wifi_config() {
    Preferences prefs;
    if (prefs.begin("gauge_wifi", false)) {
        prefs.remove("ssid");
        prefs.remove("pass");
        prefs.end();
    }
    s_wifi_ssid = "";
    s_wifi_pass = "";
}

static void connect_wifi() {
    if (s_wifi_ssid.length() == 0) {
        Serial.println("未配置 WiFi：串口输入 WIFI SET <ssid> <pass>");
        return;
    }
    WiFi.mode(WIFI_STA);
    WiFi.begin(s_wifi_ssid.c_str(), s_wifi_pass.c_str());
    Serial.printf("正在连接 WiFi %s ...\n", s_wifi_ssid.c_str());
    unsigned long wifi_t0 = millis();
    while (WiFi.status() != WL_CONNECTED &&
           millis() - wifi_t0 < WIFI_TIMEOUT_MS) {
        delay(200);
    }
    if (WiFi.status() == WL_CONNECTED) {
        MDNS.begin(STA_HOST);
        MDNS.addService("http", "tcp", 80);
        Serial.printf("STA: %s, IP: http://%s/  (mDNS: http://%s.local/)\n",
                      s_wifi_ssid.c_str(), WiFi.localIP().toString().c_str(),
                      STA_HOST);
    } else {
        Serial.printf("WiFi 连接失败（%s），仅串口可用\n", s_wifi_ssid.c_str());
    }
}

// ==================== 核心读数 ====================
static bool angle_in_arc(float a, float min_a, float max_a, int sweep) {
    if (sweep == 0) {
        return gauge::relative_angle(min_a, a) <=
               gauge::relative_angle(min_a, max_a);
    }
    return gauge::relative_angle(a, min_a) <=
           gauge::relative_angle(max_a, min_a);
}

static Reading compute_reading() {
    Reading r;
    r.ts_ms = millis();
    const Calibration& cal = get_calibration();
    if (!cal.valid) {
        r.error = "not calibrated (serial: CAL CENTER AUTO -> CAL MIN -> CAL MAX -> CAL SAVE)";
        return r;
    }

    int w = 0, h = 0;
    const uint8_t* gray = capture_grayscale(&w, &h);
    if (!gray) {
        r.error = "capture failed";
        return r;
    }

    gauge::ScanResult scan =
        gauge::scan_needle_angle(gray, w, h, cal.cx, cal.cy, cal.radius);
    if (!scan.ok) {
        r.error = "needle not found";
        return r;
    }

    float angle = gauge::normalize_angle(scan.angle_deg);
    // 180° 消歧：优先使指针落在标定量程弧内
    if (!angle_in_arc(angle, cal.min_angle, cal.max_angle, cal.sweep)) {
        angle = gauge::normalize_angle(angle + 180.0f);
    }

    float ratio = gauge::calculate_ratio(cal.min_angle, cal.max_angle,
                                         angle, cal.sweep);
    if (ratio < 0.0f) {
        r.error = "invalid scale span";
        return r;
    }
    r.ok = true;
    r.bar = gauge::calculate_value(cal.bar_min, cal.bar_max, ratio);
    r.psi = gauge::calculate_value(cal.psi_min, cal.psi_max, ratio);
    r.ratio = ratio;
    r.angle = angle;
    r.conf = scan.confidence;
    return r;
}

static String reading_json(const Reading& r) {
    String s = "{";
    s += "\"ok\":" + String(r.ok ? "true" : "false");
    s += ",\"ts\":" + String(r.ts_ms);
    if (r.ok) {
        s += ",\"bar\":" + String(r.bar, 1);
        s += ",\"psi\":" + String(r.psi, 1);
        s += ",\"ratio\":" + String(r.ratio, 4);
        s += ",\"angle\":" + String(r.angle, 1);
        s += ",\"conf\":" + String(r.conf, 3);
    } else {
        s += ",\"error\":\"" + r.error + "\"";
    }
    s += "}";
    return s;
}

// ==================== Web 路由 ====================
static void handle_root() {
    const Calibration& cal = get_calibration();
    String page = R"rawliteral(<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gauge AI Reader</title>
<style>body{font-family:Arial;background:#111;color:#eee;margin:0;padding:16px}
.card{background:#1b1b1b;border-radius:12px;padding:16px;margin-bottom:12px}
.big{font-size:36px;font-weight:bold}.dim{color:#888;font-size:13px}
button{background:#2e7d32;color:#fff;border:0;border-radius:8px;padding:10px 16px;font-size:16px}
img{width:100%;border-radius:8px}</style></head><body>
<div class="card"><h2>指针式仪表读数</h2>
<div id="r" class="dim">加载中...</div></div>
<div class="card"><button onclick="location.href='/capture'">查看画面</button>
<div class="dim" style="margin-top:8px">标定状态: )rawliteral";
    page += cal.valid ? "已标定" : "<b style='color:#f44336'>未标定</b>";
    page += R"rawliteral(</div></div>
<script>
async function refresh(){
  const el=document.getElementById('r');
  try{
    const j=await (await fetch('/reading')).json();
    if(j.ok){el.innerHTML='<div class="big">'+j.bar+' bar</div>'+
      '<div>psi: '+j.psi+' &nbsp; ratio: '+(j.ratio*100).toFixed(1)+'%</div>'+
      '<div class="dim">angle: '+j.angle+'&deg; conf: '+j.conf+' ts: '+j.ts+'ms</div>';}
    else{el.innerHTML='<div class="big" style="color:#f44336">未读数</div>'+
      '<div class="dim">'+j.error+'</div>';}
  }catch(e){el.textContent='网络错误: '+e.message;}
}
refresh(); setInterval(refresh, 1000);
</script></body></html>)rawliteral";
    server.send(200, "text/html; charset=utf-8", page);
}

static void handle_reading() {
    server.send(200, "application/json", reading_json(s_reading));
}

static void handle_capture() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        server.send(503, "text/plain", "capture failed");
        return;
    }
    uint8_t* jpg = nullptr;
    size_t len = 0;
    bool ok = (fb->format == PIXFORMAT_JPEG)
                  ? (jpg = fb->buf, len = fb->len, true)
                  : frame2jpg(fb, 80, &jpg, &len);
    if (!ok || !jpg) {
        esp_camera_fb_return(fb);
        server.send(500, "text/plain", "jpeg encode failed");
        return;
    }
    server.sendHeader("Cache-Control", "no-store");
    server.setContentLength(len);
    server.send(200, "image/jpeg", "");
    server.client().write(jpg, len);
    if (fb->format != PIXFORMAT_JPEG) free(jpg);
    esp_camera_fb_return(fb);
}

// ==================== 串口菜单 ====================
static void serial_help() {
    Serial.println("--- 命令菜单 ---");
    Serial.println("READ                    手动读一次");
    Serial.println("CAL CENTER AUTO         表盘中心=画面中心，半径=min(w,h)*0.45");
    Serial.println("CAL CENTER x y r        手动指定中心与半径");
    Serial.println("CAL MIN [F]             记录当前指针角度为 min（F=翻转180°）");
    Serial.println("CAL MAX [F]             记录当前指针角度为 max（F=翻转180°）");
    Serial.println("CAL SWEEP cw|ccw        扫掠方向（从 min 到 max 的走法）");
    Serial.println("CAL SHOW                 显示当前标定");
    Serial.println("CAL SAVE                 保存标定到 NVS");
    Serial.println("CAL CLEAR                清除标定");
    Serial.println("WIFI SET <ssid> <pass>  保存 WiFi 配置并连接（密码存 NVS）");
    Serial.println("WIFI SHOW                显示已配置的 WiFi（密码打码）");
    Serial.println("WIFI CLEAR               清除 WiFi 配置");
    Serial.println("HELP                     显示本菜单");
}

static void process_wifi_command(const String& raw) {
    String cmd = raw.substring(5);
    cmd.trim();
    String up = cmd;
    up.toUpperCase();
    if (up == "SHOW") {
        if (s_wifi_ssid.length() == 0) {
            Serial.println("未配置 WiFi");
        } else {
            Serial.printf("ssid=%s pass=%s（已存 NVS）\n",
                          s_wifi_ssid.c_str(),
                          s_wifi_pass.length() > 0 ? "****" : "");
        }
        return;
    }
    if (up == "CLEAR") {
        clear_wifi_config();
        WiFi.disconnect();
        Serial.println("WiFi 配置已清除");
        return;
    }
    if (up.startsWith("SET ")) {
        String rest = cmd.substring(4);
        rest.trim();
        int sp = rest.indexOf(' ');
        if (sp <= 0) {
            Serial.println("用法: WIFI SET <ssid> <pass>");
            return;
        }
        String ssid = rest.substring(0, sp);
        String pass = rest.substring(sp + 1);
        pass.trim();
        save_wifi_config(ssid, pass);
        Serial.printf("已保存 WiFi 配置: %s\n", ssid.c_str());
        connect_wifi();
        return;
    }
    Serial.println("未知 WiFi 命令（HELP 查看）");
}

static bool scan_angle_now(float* out_angle, float* out_conf) {
    int w = 0, h = 0;
    const uint8_t* gray = capture_grayscale(&w, &h);
    if (!gray) return false;
    const Calibration& cal = get_calibration();
    float cx = cal.valid ? cal.cx : w * 0.5f;
    float cy = cal.valid ? cal.cy : h * 0.5f;
    float radius = cal.valid ? cal.radius : min(w, h) * DEFAULT_RADIUS_RATIO;
    gauge::ScanResult s = gauge::scan_needle_angle(gray, w, h, cx, cy, radius);
    if (!s.ok) return false;
    *out_angle = s.angle_deg;
    *out_conf = s.confidence;
    return true;
}

static void process_serial_line(String line) {
    line.trim();
    if (line.length() == 0) return;
    String line_up = line;
    line_up.toUpperCase();
    if (line_up.startsWith("WIFI ")) {   // 密码大小写敏感，须在 toUpperCase 前处理
        process_wifi_command(line);
        return;
    }
    line = line_up;
    if (line == "HELP") { serial_help(); return; }
    if (line == "READ") {
        s_reading = compute_reading();
        Serial.println(reading_json(s_reading));
        return;
    }
    if (line.startsWith("CAL ")) {
        Calibration cal = get_calibration();
        if (line == "CAL CENTER AUTO") {
            int w = 0, h = 0;
            const uint8_t* gray = capture_grayscale(&w, &h);
            if (!gray) { Serial.println("capture failed"); return; }
            cal.cx = w * 0.5f;
            cal.cy = h * 0.5f;
            cal.radius = min(w, h) * DEFAULT_RADIUS_RATIO;
            set_calibration(cal);
            Serial.printf("center=%.0f,%.0f radius=%.0f (w=%d h=%d)\n",
                          cal.cx, cal.cy, cal.radius, w, h);
            return;
        }
        if (line.startsWith("CAL CENTER ")) {
            float x, y, r;
            if (sscanf(line.c_str() + 11, "%f %f %f", &x, &y, &r) == 3) {
                cal.cx = x; cal.cy = y; cal.radius = r;
                set_calibration(cal);
                Serial.println("center set");
            } else {
                Serial.println("用法: CAL CENTER <x> <y> <r>");
            }
            return;
        }
        if (line == "CAL MIN" || line == "CAL MIN F" ||
            line == "CAL MAX" || line == "CAL MAX F") {
            float a = 0, conf = 0;
            if (!scan_angle_now(&a, &conf)) {
                Serial.println("未检测到指针，请确认表盘在画面内且指针清晰");
                return;
            }
            bool flip = line.endsWith("F");
            if (flip) a = gauge::normalize_angle(a + 180.0f);
            if (line.startsWith("CAL MIN")) cal.min_angle = a;
            else cal.max_angle = a;
            set_calibration(cal);
            Serial.printf("recorded %s=%.1f (conf=%.2f)\n",
                          line.startsWith("CAL MIN") ? "min" : "max", a, conf);
            return;
        }
        if (line == "CAL SWEEP CW") { cal.sweep = 0; set_calibration(cal); Serial.println("sweep=cw"); return; }
        if (line == "CAL SWEEP CCW") { cal.sweep = 1; set_calibration(cal); Serial.println("sweep=ccw"); return; }
        if (line == "CAL SHOW") {
            const Calibration& c = get_calibration();
            if (!c.valid) { Serial.println("未标定"); return; }
            Serial.printf("center=%.0f,%.0f r=%.0f min=%.1f max=%.1f sweep=%s bar=%.1f~%.1f psi=%.1f~%.1f\n",
                          c.cx, c.cy, c.radius, c.min_angle, c.max_angle,
                          c.sweep ? "ccw" : "cw", c.bar_min, c.bar_max,
                          c.psi_min, c.psi_max);
            return;
        }
        if (line == "CAL SAVE") { save_calibration(); Serial.println("已保存"); return; }
        if (line == "CAL CLEAR") { clear_calibration(); Serial.println("已清除"); return; }
        Serial.println("未知标定命令（HELP 查看）");
        return;
    }
    Serial.println("未知命令（HELP 查看）");
}

// ==================== 启动与主循环 ====================
void setup() {
    Serial.begin(115200);
    Serial.setDebugOutput(true);
    delay(1500);

    Serial.println("\n=== Gauge AI Reader (ESP32-S3, 传统法 v1) ===");
    WiFi.onEvent(on_wifi_event, ARDUINO_EVENT_WIFI_STA_DISCONNECTED);
    if (load_calibration()) {
        Serial.println("已加载标定");
    } else {
        Serial.println("无标定，请先完成标定流程（HELP 查看）");
    }

    if (!init_camera()) {
        Serial.println("摄像头初始化失败，系统暂停");
        while (1) delay(1000);
    }
    optimize_camera_image();

    if (STA_SSID[0] != '\0') {
        // 本地写死凭据（firmware/src/wifi_credentials.h，不入库）
        s_wifi_ssid = STA_SSID;
        s_wifi_pass = STA_PASS;
        connect_wifi();
    } else if (load_wifi_config()) {
        connect_wifi();
    } else {
        Serial.println("未配置 WiFi：串口输入 WIFI SET <ssid> <pass>");
    }

    server.on("/", handle_root);
    server.on("/reading", handle_reading);
    server.on("/capture", handle_capture);
    server.begin();

    serial_help();
}

void loop() {
    if (WiFi.status() != WL_CONNECTED) {
        static unsigned long last_wifi_try = 0;
        if (millis() - last_wifi_try >= 5000) {
            last_wifi_try = millis();
            WiFi.reconnect();
        }
    }
    server.handleClient();

    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n') {
            process_serial_line(s_serial_buf);
            s_serial_buf = "";
        } else if (c != '\r') {
            s_serial_buf += c;
        }
    }

    unsigned long now = millis();
    if (now - s_last_read_ms >= READ_INTERVAL_MS) {
        s_last_read_ms = now;
        s_reading = compute_reading();
    }
}
