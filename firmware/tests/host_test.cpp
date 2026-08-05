// 主机回归测试：验证 radial_scan / geometry 的 C++ 移植与 Python 版一致。
// 编译（项目根目录，需 g++）：
//   g++ -std=c++17 -O2 -Ifirmware/src firmware/tests/host_test.cpp \
//       firmware/src/radial_scan.cpp -o host_test.exe

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "geometry.h"
#include "radial_scan.h"

static int s_fail = 0;

static void check(bool cond, const char* name) {
    if (cond) {
        printf("  PASS  %s\n", name);
    } else {
        printf("  FAIL  %s\n", name);
        ++s_fail;
    }
}

static bool load_pgm(const char* path, std::vector<uint8_t>& out,
                     int& w, int& h) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    char magic[3] = {0};
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) {
        fclose(f);
        return false;
    }
    if (fscanf(f, "%d %d", &w, &h) != 2) { fclose(f); return false; }
    int maxv = 0;
    if (fscanf(f, "%d", &maxv) != 1) { fclose(f); return false; }
    fgetc(f);
    out.resize((size_t)w * h);
    if (fread(out.data(), 1, out.size(), f) != out.size()) {
        fclose(f);
        return false;
    }
    fclose(f);
    return true;
}

static float angle_diff(float a, float b) {
    float d = fmodf(fabsf(a - b) + 180.0f, 360.0f) - 180.0f;
    return fabsf(d);
}

int main() {
    const char* img_dir = "firmware/tests/images/";
    const int cases[] = {45, 90, 135, 200, 300};
    printf("== 径向扫描（合成表盘，center=200,200 r=180）==\n");
    for (int expect : cases) {
        char path[256];
        snprintf(path, sizeof(path), "%sgauge_%d.pgm", img_dir, expect);
        std::vector<uint8_t> gray;
        int w = 0, h = 0;
        if (!load_pgm(path, gray, w, h)) {
            printf("  SKIP  %s (缺测试图，先运行 gen_test_images.py)\n", path);
            continue;
        }
        gauge::ScanResult r =
            gauge::scan_needle_angle(gray.data(), w, h, 200.0f, 200.0f, 180.0f);
        char name[96];
        snprintf(name, sizeof(name), "needle=%d -> angle=%.2f conf=%.3f",
                 expect, r.angle_deg, r.confidence);
        check(r.ok && angle_diff(r.angle_deg, (float)expect) < 3.0f, name);
    }

    printf("== 空场景（纯白，应 not ok）==\n");
    {
        std::vector<uint8_t> gray;
        int w = 0, h = 0;
        if (load_pgm("firmware/tests/images/blank.pgm", gray, w, h)) {
            gauge::ScanResult r =
                gauge::scan_needle_angle(gray.data(), w, h, 200.0f, 200.0f, 180.0f);
            check(!r.ok, "blank -> not ok");
        }
    }

    printf("== geometry ==\n");
    check(fabsf(gauge::normalize_angle(-45.0f) - 315.0f) < 1e-4f,
          "normalize(-45)=315");
    check(fabsf(gauge::relative_angle(350.0f, 10.0f) - 20.0f) < 1e-4f,
          "relative(350,10)=20");
    check(fabsf(gauge::calculate_ratio(0, 90, 45, 0) - 0.5f) < 1e-4f,
          "ratio cw(0,90,45)=0.5");
    check(fabsf(gauge::calculate_ratio(0, 90, 180, 1) -
                (180.0f / 270.0f)) < 1e-4f,
          "ratio ccw(0,90,180)=2/3");
    check(gauge::calculate_ratio(0, 90, 45, -1) == -1.0f, "invalid sweep -> -1");
    check(fabsf(gauge::calculate_value(0, 250, 0.5f) - 125.0f) < 1e-4f,
          "value(0,250,0.5)=125");

    printf(s_fail == 0 ? "\nALL PASS\n" : "\n%d FAILED\n", s_fail);
    return s_fail == 0 ? 0 : 1;
}
