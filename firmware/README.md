# ESP32-S3 指针表读数固件（传统法 v1）

基于 **TFLite-free 的传统方法**：OV2640 采集 → RGB565 → 灰度 →
径向扫描检测指针角度 → 双量程读数。DL 关键点模型（路线 B）后续再集成。

## 硬件

- ESP32-S3-WROOM-1 N16R8（16MB Flash / 8MB PSRAM）
- OV2640 摄像头（针脚定义与 `my_first_esp32s3` 旧工程一致）

## 构建与烧录（VS Code + PlatformIO）

1. 用 VS Code 打开 `firmware/` 目录（PlatformIO 插件）；
2. 上传：`PlatformIO: Upload`；打开串口监视器（115200）看日志；
3. 手机/电脑连接 WiFi `ESP32S3-GAUGE`（密码 `12345678`），
   浏览器打开 `http://192.168.4.1/` 查看实时读数与画面。

## 标定流程（固定安装后只需做一次，存 NVS）

把摄像头对准表盘（尽量让表盘居中、占画面 60% 以上），串口依次执行：

```text
CAL CENTER AUTO        表盘中心=画面中心（不理想就用 CAL CENTER x y r 手动指定）
<手动把指针拨到刻度起点> CAL MIN
<手动把指针拨到刻度终点> CAL MAX
CAL SWEEP cw|ccw       按表盘从 min 走到 max 的实际方向设置
CAL SHOW               检查标定值
CAL SAVE               保存（断电不丢）
```

如果读数方向反了（指针越走数值越小），用 `CAL MIN F` / `CAL MAX F`
重新记录翻转后的角度，或调整 `CAL SWEEP`。

## 串口命令

`READ`、`CAL CENTER AUTO`、`CAL CENTER <x> <y> <r>`、`CAL MIN [F]`、
`CAL MAX [F]`、`CAL SWEEP cw|ccw`、`CAL SHOW`、`CAL SAVE`、`CAL CLEAR`、`HELP`

## Web 接口

- `GET /` 读数页面（每秒自动刷新）
- `GET /reading` JSON：`{"ok":true,"bar":..,"psi":..,"ratio":..,"angle":..,"conf":..}`
- `GET /capture` 当前画面 JPEG

## 双量程

与 `config/gauge.yaml` 一致：bar 0~250，psi 0~3625.9，按同一比例线性映射
（修改 `src/config.h` 中的量程宏）。

## 算法说明

`src/radial_scan.cpp` 是 Python `src/refiner.py` 径向扫描的 C++ 移植：
环带 Otsu 阈值（5/95 分位钳制）→ 双极性“从内缘连续暗段”扫描 →
有效得分选极性 → 抛物线亚度插值。`src/geometry.h` 对应 `geometry.py`。
算法已在主机上用合成表盘图做过回归验证（`tests/host_test.cpp`）。

## 已知限制（v1）

- 表盘中心/半径靠标定（自动=画面中心或手动指定），固定安装够用；
- 单表盘、单量程几何（min/max 两处标定角度），不支持画面内多表；
- 强反光/背光会降低径向扫描置信度，`conf` 低于阈值时返回 `needle not found`；
- 角度比例法对斜拍视角敏感，摄像头应尽量正对表盘。
