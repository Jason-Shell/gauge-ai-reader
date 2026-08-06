# ESP32-S3 指针表读数固件（传统法 v1）

基于 **TFLite-free 的传统方法**：OV2640 采集 → RGB565 → 灰度 →
径向扫描检测指针角度 → 双量程读数。DL 关键点模型（路线 B）后续再集成。

## 硬件

- ESP32-S3-WROOM-1 N16R8（16MB Flash / 8MB PSRAM）
- OV2640 摄像头（针脚定义与 `my_first_esp32s3` 旧工程一致）

## 构建与烧录（VS Code + PlatformIO）

1. 用 VS Code 打开 `firmware/` 目录（PlatformIO 插件）；
2. 上传：`PlatformIO: Upload`；打开串口监视器（115200）看日志；
3. 设备以 STA 模式连入局域网，手机/电脑浏览器打开启动日志里打印的
   IP（或 `http://gauge.local/`）查看实时读数与画面。

## 连接与访问（STA 模式，无 AP）

两种配网方式，按优先级：

1. **本地写死（推荐本机自用，密码不入库）**：把
   `firmware/src/wifi_credentials.example.h` 复制为
   `firmware/src/wifi_credentials.h` 并填入真实 SSID / 密码。
   `wifi_credentials.h` 已被 `.gitignore` 排除，不会提交到仓库；
2. **串口配网（不创建上面的文件时生效）**：首次烧录后用串口命令配置，
   密码只存设备 NVS，`WIFI SHOW` 打码显示：

```text
WIFI SET <ssid> <pass>   保存并连接（例：WIFI SET MyWiFi MyPassword）
WIFI SHOW                查看已配置 SSID（密码打码）
WIFI CLEAR               清除配置
```

- 烧录后看串口日志：连接成功会打印 `STA: <ssid>, IP: http://192.168.x.x/`
  （`http://gauge.local/` 为 mDNS 别名）；连接失败（超时 `WIFI_TIMEOUT_MS`）
  则仅串口可用，程序不阻塞；
- 路由器重启导致 IP 变化时，`loop()` 每 5 秒自动重连，重连后 mDNS 随
  新 IP 更新；
- `STA_HOST`（mDNS 主机名）与 `WIFI_TIMEOUT_MS` 在 `src/config.h` 中，
  改后需重新编译；串口配网时 SSID / 密码写入设备 NVS，不进入代码仓库。

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

## USB 串口直读（无 WiFi 方案）

不需要 WiFi / 浏览器：固件每 `READ_INTERVAL_MS`（默认 500ms）自动在
USB 串口打印一条读数 JSON，串口监视器（115200）里直接看：

```json
{"ok":true,"ts":123456,"bar":101.5,"psi":1472.1,"ratio":0.4060,"angle":152.3,"conf":0.812}
```

未标定时打印 `{"ok":false,"error":"not calibrated ..."}`。WiFi 正常时
网页 `/reading` 与串口输出并存，互不影响。

## Web 接口

- `GET /` 读数页面（每秒自动刷新）
- `GET /reading` JSON：`{"ok":true,"bar":..,"psi":..,"ratio":..,"angle":..,"conf":..}`
- `GET /capture` 当前画面 JPEG

批量抓图可配合仓库根目录的 `scripts/capture_images.py`（PC 端定时
拉取 `/capture` 存为 JPEG，供之后标注读数）。

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
