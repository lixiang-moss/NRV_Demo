# NRV_Demo

[English](README.md) | **简体中文**

NRV DELTA01 的 E2FAI 重建与光流演示，以 `12fe7f9` 为工程基线。ROS、Qt 与官方原图渲染运行在容器内；E2FAI 运行在宿主机 GPU 进程，通过本机 TCP 与 C++ ROS 桥通信。原 Python 质心算法和软件过滤器已移除。

## 启动

宿主机需要 Linux x86_64、Docker Engine / Compose v2、Conda、支持 CUDA 12.1 PyTorch 构建的 NVIDIA 驱动；GUI 需要 X11 或 XWayland、`DISPLAY` 和 `xhost`。先停止占用相机的其他程序。

```bash
./scripts/setup_e2fai.sh  # 首次创建宿主模型环境
./scripts/build.sh      # 首次构建或修改容器内代码后重建
./scripts/run.sh
```

环境为 Python 3.10、PyTorch 2.1.1、NumPy 1.26.4、OpenCV 4.7.0.72。启动器默认使用 `~/miniconda3/envs/nrv-e2fai/bin/python`，可用 `E2FAI_PYTHON` 指定其他位置。配套权重放在 `checkpoints/e2fai_backbone.ckpt` 和 `checkpoints/image_residual_epoch043.pt`；安装脚本不下载权重。

## 画面与相机参数

默认显示三幅画面：**Original rendering**（官方事件原图）、**E2FAI reconstruction**（重建灰度图）、**E2FAI optical flow**（光流彩色预览）。在 **Views** 选择 1–3 幅；一幅占满，两幅并排，三幅按两列排列。隐藏画面只影响显示，不停止模型或改变输入。

图像控件不再用收到的 Pixmap 尺寸决定布局，因此异步到图不会挤动其他画面；主窗口仍可自由缩放。

**Settings** 中保留 ON/OFF 硬件参数。编辑后点击 **Apply parameters** 生效；修改硬件参数会重启采集并创建新模型会话。ON/OFF 是寄存器 `0x0167` / `0x0168` 的低 6 位十进制编码，范围 0–63，不是标定后的灵敏度。GUI 保留源配置的其他位和参数。硬件参数影响原图与模型输入。

**Save profile / Load profile** 保存相机参数和画面选择。加载相机参数后仍须 Apply；画面选择立即生效。旧配置的 `filters` 被忽略，旧 `algorithm` 画面被移除；如果没有剩余画面，恢复默认三幅。自动保存位置为 `output/last_applied_parameters.json`。

**Stop** 停止采集及当前推理会话、清空循环状态，保留已加载权重。**Start** 创建新会话，旧结果不会进入新画面。关闭 GUI 或 Ctrl+C 后，启动器退出 GPU 进程并释放端口。

## 模型与时间尺度

默认使用 **250 ms 半开窗口 `[start, end)`、960×720 原生分辨率、全部事件和 15-bin 体素**。保留 U-Net、光流池化与插值、ConvGRU 图像残差及两份权重。模型直接读取 `/dvs/events`；不排序、不插值替代真实事件时间、不抽样。

`32FC2` 光流单位是 **推理网格像素／输入窗口**，当前默认即像素／250 ms；若需平均像素速度，用光流除以实际窗口秒数。彩色预览默认饱和尺度为 **50 像素／250 ms**，对应 200 像素/秒，仅改变显示颜色，不改变浮点光流。跨窗循环状态在时间倒退、超过 **300 ms** 的真实事件断流、尺寸/连接变化或检测到批次间断时清理。输入队列超限会暂停会话并报告，Stop → Start 可重新开始。

默认 `E2FAI_RESULT_MODE=thread`：CPU 结果转图与 TCP 回传由独立线程按顺序执行，可与后续推理重叠；保留有界队列和背压，不丢弃窗口。设置 `E2FAI_RESULT_MODE=inline` 可恢复串行结果处理，便于对照或撤回。

宿主输入队列默认最多 **64 个 ROS 事件批次或 1 GiB（1024 MiB）**，先达到者生效；这不是 64 个推理窗口。用 `E2FAI_QUEUE_BATCHES=128 ./scripts/run.sh` 可调整批数，设为 `32` 可恢复之前的上限。容器 C++ 发送队列保持 32 批。加长队列可容纳更多突发，但不会提高模型处理速度，可能增加显示延迟。已有真机报告使用的是此前 256 MiB 上限；本次 1 GiB 扩容通过队列边界检查，未追加真机测试。

解码时间保留逐事件顺序及传感器计数关系；约 4,295 秒的异常跳变在解码层修复，不用包头插值掩盖。映射事件时间计算出的年龄不等于标定后的物理端到端延迟。

## 常用命令

```bash
# 必要的短时真机检查；结束采集后 GUI 保留
DURATION=20 ./scripts/run.sh

# 无界面，仍运行模型
SHOW_GUI=false DURATION=20 ./scripts/run.sh

# 开启周期性能记录（默认关闭）
PERF_ENABLED=true DURATION=20 ./scripts/run.sh

# 只看官方原图，不启动宿主模型
E2FAI_ENABLED=false ./scripts/run.sh

# 只接收 RAW，不解码、推理或渲染
DECODE_EVENTS=false SHOW_GUI=false DURATION=20 ./scripts/run.sh

# 指定相机或覆盖窗口时长
CAMERA_INDEX=1 ./scripts/run.sh
WINDOW_MS=250 ./scripts/run.sh

# 撤回结果线程，保留 250 ms 窗口
E2FAI_RESULT_MODE=inline ./scripts/run.sh
```

## 数据链路与结果

```mermaid
flowchart LR
  Camera[DELTA01] --> Driver[官方驱动]
  Driver -->|RAW| Renderer[官方渲染]
  Driver -->|RAW| Adapter[逐事件解码]
  Driver -->|RAW| Monitor[采集计数]
  Adapter -->|/dvs/events| Bridge[C++ ROS桥]
  Bridge <-->|本机TCP| Model[宿主GPU E2FAI]
  Renderer --> GUI[Qt 1–3幅画面]
  Bridge -->|/nrv_e2fai/result| GUI
  Monitor -->|/nrv_capture/status| GUI
```

驱动、解码器、渲染器共享 nodelet manager 进程；采集计数、C++ 桥和 GUI 各自独立。TCP 默认监听 `127.0.0.1:8765`。Qt 使用最新图像缓存，不等待推理完成。

| 话题 | 内容 |
| --- | --- |
| `/delta_driver/events` | `event_camera_msgs/EventPacket`：RAW 字节及原始元数据 |
| `/dvs/events` | `dvs_msgs/EventArray`：按顺序的 `(x, y, ts, polarity)` |
| `/delta_renderer/image` | 官方原图 |
| `/nrv_e2fai/result` | 会话/窗口/来源批次元数据、`mono8` 灰度、`rgb8` 光流预览和 `32FC2` 光流 |
| `/nrv_e2fai/status` | 桥连接、暂停状态和计数 |
| `/nrv_capture/status` | RAW 包数、字节率和序号间断次数 |

运行结果在 `output/run_*/`。`summary.json` 只检查 RAW 是否接收及序号连续性；`raw_sequence_gaps` 是间断**次数**。模型和桥分别保存 `e2fai_session_*.json`、`sessions/<session>/bridge_summary.json`。`integration_summary.json` 检查 RAW、模型出图、会话错误与 ROS 批次间断；合法时间重置单独记录，不自动判失败。PASS 不代表实时性能验收。设置 `PERF_ENABLED=true` 时另写周期 `performance_*.jsonl`。GUI 显示时间测量截止 `setPixmap`，不是显示器实际呈现时刻。

当前结果见 [E2FAI 250 ms 与结果线程实测](docs/E2FAI250ms与结果线程实测.md)，以 Markdown 交付。之前的[E2FAI 修复与真机短测](docs/E2FAI修复与真机短测.md)和[移植后性能测试与瓶颈分析](docs/移植后性能测试与瓶颈分析.md)保留为历史记录，其中旧窗口时长与旧桥接实现的数据不代表当前版本性能。

[examples/raw_receiver.py](examples/raw_receiver.py) 保留为独立 RAW 接收示例，不在演示链路中运行。`group_aer` 是带跨包状态的编码数据，不是完整 `.dvs` 文件；下游需保留编码、序号、时间基准、宽高等元数据。

镜像依赖 Ubuntu 20.04、ROS Noetic 与 NRV SDK/驱动。修改容器内代码后执行 `./scripts/build.sh`。USB 通过 `/dev/bus/usb` 提供，启动器临时授予容器 X11 权限并在退出时撤销。代码采用 [MIT](LICENSE)；厂商依赖和 `dvs_msgs` 按各自许可使用。
