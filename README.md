# entopo VLM Navigation

本项目当前主要演进方向为端到端的 VLM 导航智能体，核心流程为：
`RGB (包含连续帧) -> 地面扇形投影 (A~E) -> VLM 评估安全与选向 -> 执行物理反馈动作 -> 记录轨迹/地图`

## 决策循环（Decision Loop）

每个 episode 内的导航循环如下：

1. **环境观测**：读取当前仿真的 RGB 图片，并保存连续的无修改原始帧用于回放。
2. **两阶段评估**：
   - **Stage 1 (安全/死胡同检查)**：询问 VLM 前方是否有任何安全通过的可能（输出 Y/N）。如果不安全，立即触发后退或旋转的 Fallback 动作。
   - **Stage 2 (方向选择)**：在有效行驶空间的地板上绘制 A~E 5条等长的短轨迹线。调用 VLM 选择最安全的行进路线。
3. **熵安全网 (Entropy Guard)**：解析 VLM 对各个选项的概率。如果最大置信度极低（< 0.35）或概率分布的熵值过高（> 2.1），说明 VLM 在瞎猜，此时剥夺执行权，触发 Fallback 动作。
4. **动作执行**：把通过安全网的选项分解为底层的 primitive action (比如数次 `turn_left` 和 `move_forward`) 并连续执行。
5. **记录日志**：记录碰撞情况、位姿轨迹以及各选项概率，循环至抵达目标或达到步数上限。

## 动作空间与映射 (A~E)

VLM 只需输出一个字母，代表其选择的宏观方向：

- `A`: 左大转（如左转两次）然后前进
- `B`: 左微转（左转一次）然后前进
- `C`: 直行前进
- `D`: 右微转（右转一次）然后前进
- `E`: 右大转（如右转两次）然后前进

*备注：当触发 Fallback 或检测到碰撞时，智能体会执行特殊的纠正动作（如单纯的原地向左旋转 30° 探路），这个过程通过代码后置处理，不需要通过 VLM 输出独立的 Token。*

## 地面轨迹投影 (Ground Overlay)

- **告别 HUD 箭头**：放弃了容易受透视变形影响的屏幕屏幕贴片，改用直接在 3D 物理空间的地面渲染。
- 以相机中心为原点，向外绘制 5 条等长、呈扇形展开的轨迹线，完美对齐实际的偏航角 (yaw: -60, -30, 0, 30, 60) 和步长。
- 轨迹线的末端画有高对比度实心圆作为锚点，标有利字母 A~E。

## Habitat 主要配置

- **RGB ONLY**：当前模型纯依靠 RGB 图像输入操作，不再依赖 Depth 传感器。
- **动态视高**：可以在 `configs/vlm_nav.json` 中配置 `camera_height_m` 从而物理改变机器人的基础身高与视角，有效避免被脚下的家具和墙根卡住。
- **灵活的物理参数**：每次执行的 `forward_step_m` 和 `scan_angle_deg` 均在 JSON 中配置并生效。

## 目录与文件核心结构

- `vlm_nav/actions.py`：动作 (A-E) 的定义与到底层 primitive 操作的映射。
- `vlm_nav/ground_overlay.py`：基于地面的真实物理空间扇形轨迹重绘逻辑。
- `vlm_nav/vlm_client.py`：VLM API 请求、包含 Y/N 和 A~E 两阶段语法约束配置、Prompt。
- `vlm_nav/agent.py`：系统主要引擎（端到端 `while` 循环、结合熵控与规则覆盖的安全策略提取核心）。
- `vlm_nav/config.py`：`VLMConfig`, `NavigationConfig` 和 `HabitatConfig` 数据类别管理。
- `scripts/run_habitat_agent.py`：**端到端仿真跑查主入口**。
- `configs/vlm_nav.json`：全局运行参数文件。

## 运行示例

### 1) 启动 VLM 服务（如 llama.cpp / vLLM）
确保你的大模型以兼容 OpenAI 格式的 API 服务拉起（必须支持 `grammar` 与 `logprobs` 参数）。

### 2) 运行端到端 Habitat 导航的主线

```bash
conda run -n habitat python scripts/run_habitat_agent.py \
  --config configs/vlm_nav.json \
  --run-name mainline_run \
  --max-episodes 4 \
  --max-steps 500 \
  --vlm-seed 3407
```
*如需固定 VLM 的 seed 做对照测试，可以增加 `--vlm-seed 3407` 参数。*

### 3) 运行 mini-dataset (针对离线图像集的纯推理与渲染测试)

```bash
conda run -n habitat python scripts/run_minidataset.py \
  --config configs/vlm_nav.json \
  --limit 20 \
  --save-images
```

## 输出产物说明

执行完成后，在 `outputs/<run-name>/` 目录下将会得到极其丰富的可观测结果：

- `summary.csv`, `summary.json`：回合级别的总体统计（成功率、平均熵、退回次数等）。
- `ep*.jsonl`：逐次 Decision 的明细日志（选了什么选项、算力消耗、碰撞触发次数、熵和后验概率）。
- `ep*_trace.csv`：机器人每个物理步进的真实的 3D 环境空间坐标。
- `ep*_topdown.png`：当前环境的俯视地图（自动绘制出行走路线）。
- `ep*_overlays/`：VLM 眼中看到的（已经画上了 A~E 地标）的推理图集。
- `ep*_prob_vis/`：在渲染上方附加了详细选项置信度的统计面板图。
- `ep*_frames/`：**包含每一帧（包括多 primitive 里无法推理决策时的空档过渡帧）的完整连续视像，用于无缝回放**。
