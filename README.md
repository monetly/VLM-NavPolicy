# VLM-NavPolicy (OmniSight-Nav)
[English](#english) | [中文](#chinese)

<a name="english"></a>
## English Documentation

### Project Overview
VLM-NavPolicy is an end-to-end autonomous navigation policy powered entirely by Vision-Language Models (VLMs). It operates strictly on RGB observations without relying on depth sensors or pre-built maps. The policy interprets the current visual observation overlaid with geometrically accurate, ground-projected trajectory candidates (A-E) and outputs a discrete semantic decision to direct the robot.

**Key Features:**
- **RGB-Only Navigation**: No depth sensor or map dependencies.
- **Physical Ground Overlay**: Action trajectories are projected into the 2D image plane using strict 3D pinhole camera perspective mathematics.
- **Grammar-Constrained Decoding**: The VLM is strictly constrained to output valid tokens (Y/N for safety checking, A-E for direction selection) with heavily analyzed log-probabilities.
- **Entropy Guard**: The system actively monitors the Shannon entropy of the VLM's probability distribution to detect hallucinations or uncertainty, triggering safety fallbacks when confidence is low.

### Architecture & Decision Loop
1. **Observation**: Capture current RGB frame.
2. **Stage 1 (Safety Check)**: The VLM is prompted with the raw image to verify if any safe, unobstructed path exists. Outputs constrained to `[Y, N]`. If `N`, fallback executed.
3. **Stage 2 (Direction Selection)**: Fan trajectories (A-E) are rendered onto the floor using physical perspective projection. VLM chooses the safest path.
4. **Entropy Guard**: Probabilities for A-E are extracted. If maximum probability $P_{max} < 0.35$ or Entropy $H > 2.1$, the decision is rejected and a fallback (e.g., rotate in place) is triggered.
5. **Execution**: The chosen semantic option (A-E) is decomposed into physical discrete actions (e.g., `turn_left` x2, `move_forward`) and executed in the simulator.

### Mathematics of 3D-to-2D Trajectory Projection
A core innovation of this project is avoiding HUD-like fixed UI overlays. Instead, trajectories are strictly mapped from the 3D floor plane to the 2D image plane using the camera's spatial parameters.

Given:
- $H_c$: Camera height in meters (e.g. `0.6m`).
- $\theta_h$: Horizontal Field of View (FOV).
- $\theta_v$: Vertical Field of View (approximated based on aspect ratio).
- $\phi_{yaw}$: Physical rotation angle resulting from the chosen action (e.g. `±30°`).
- $D_{step}$: Physical forward travel distance (m).

**1. Perspective Origin Hook ($y_{origin}$)**
Since the camera is elevated ($H_c$) and pointing forward, the physical floor directly underneath the robot ($Z=0$) is out of frame. The origin coordinate on the normalized projection plane is calculated as:
$$y_{origin} = 0.5 + 0.5 \cdot \frac{\tan(\arctan(H_c / 0))}{\tan(\theta_v / 2)} \rightarrow +\infty$$
To visualize the rays emerging from the robot's footprint, the origin is anchored slightly below the bottom edge of the image (e.g., $y_{origin} = 1.15$).

**2. Horizontal Displacement ($x_{tip}$)**
Based on the turning angle $\phi_{yaw}$, the horizontal position on the normalized projection plane is:
$$x_{tip} = 0.5 + 0.5 \cdot \frac{\tan(\phi_{yaw})}{\tan(\theta_h / 2)}$$

**3. Depth Foreshortening & Minimum Visual Ray ($y_{tip}$)**
A physical step of $0.3m$ at $H_c=0.6m$ falls entirely within the structural blind spot (visible ground starts at $\approx 0.97m$). Thus, rendering strict physical lengths would collapse all trajectories to the bottom border. 
To communicate spatial depth to the VLM, a minimum **visual ray** $D_{vis} = \max(2.5m, D_{step} \cdot 3)$ is used.
The actual depth $Z_{tip}$ on the Z-axis is projected by cosine:
$$Z_{tip} = D_{vis} \cdot \cos(\phi_{yaw})$$
The vertical tip coordinate $y_{tip}$ on the image plane is:
$$y_{tip} = 0.5 + 0.5 \cdot \frac{(H_c / Z_{tip})}{\tan(\theta_v / 2)}$$

### Quick Start
To run the full end-to-end Habitat simulation agent:
```bash
conda run -n habitat python scripts/run_habitat_agent.py \
  --config configs/vlm_nav.json \
  --run-name my_first_run \
  --max-episodes 4 --max-steps 500
```
To run the offline visual rendering tests (mini-dataset):
```bash
conda run -n habitat python scripts/run_minidataset.py \
  --config configs/vlm_nav.json \
  --limit 20 \
  --save-images
```

---

<br><br>

<a name="chinese"></a>
## 中文文档

### 项目概览
VLM-NavPolicy 是一个完全由视觉语言模型 (VLM) 驱动的端到端自主导航策略项目。该系统摒弃了深度传感器和预建地图的依赖，完全基于实时的纯 RGB 观察运行。系统会在当前视觉画面中的物理地面上绘制极其精准的扇形候选轨迹 (A-E)，由 VLM 选择安全的行进方向，从而驱动机器人移动。

**核心特性：**
- **纯视觉导航 (RGB-Only)**：无需深度图与建图模块。
- **物理空间地面投影**：所有动作轨迹严格基于 3D 针孔相机透视几何数学公式进行渲染，确保图像透视（远近、缩放、偏移）完全符合现实物理规律。
- **语法约束解码 (Grammar-Constrained Decoding)**：VLM 的输出被强制约束为有效字符（阶段一：Y/N 安全检查；阶段二：A-E 方向选择），并深度抓取输出概率矩阵 (Logprobs) 进行分析。
- **熵安全网 (Entropy Guard)**：系统实时监控 VLM 输出概率分布的香农熵 (Shannon Entropy) 和置信度。当检测到模型出现幻觉或过度不确定时，会立刻剥夺决策权并触发安全回退策略。

### 架构与决策循环
1. **环境观测**：获取当前仿真的 RGB 图片。
2. **阶段一（安全/死胡同检查）**：要求 VLM 评估前方是否有任何安全的通过可能。严格输出 `[Y, N]`。如果不安全，触发转向 Fallback。
3. **阶段二（方向选择）**：将 A-E 的轨迹投影渲染在当前画面地面上，请求 VLM 选出最安全的路线。
4. **熵安全网**：提取 A-E 的详细概率分布。若最大置信度 $P_{max} < 0.35$ 或 熵值 $H > 2.1$，判定为“瞎猜瞎走”，直接拒绝执行并触发原地旋转 Fallback。
5. **动作执行**：将选定的选项 (A-E) 翻译为底层的仿真基本动作序列（如 `turn_left` x2, `move_forward`）并执行验证。

### 3D到2D空间的轨迹投影数学原理
本项目最大的视觉创新在于：不使用贴在屏幕上的死板 2D HUD UI 箭头。所有地面的轨迹扇骨，均是通过物理相机的实际属性从 3D 世界投射到 2D 图像上的。

已知参数：
- $H_c$: 相机安装高度 (例如 `0.6` 米)。
- $\theta_h$: 水平视野角 (Horizontal FOV)。
- $\theta_v$: 垂直视野角 (Vertical FOV)。
- $\phi_{yaw}$: 该选项导致机器人在真实世界偏转的偏航角 (例如 `±30°`)。
- $D_{step}$: 机器人的真实前进步长距离 (米)。

**1. 基于视高的原点透视 ($y_{origin}$)**
当相机具有一定高度 ($H_c$) 且平视前方时，机器人脚下的正下方 ($Z=0$) 实际上位于相机的“视野盲区”内（无法被拍到）。
真正在归一化图像投影面上起算的原点坐标数学极限为：
$$y_{origin} = 0.5 + 0.5 \cdot \frac{\tan(\arctan(H_c / 0))}{\tan(\theta_v / 2)} \rightarrow +\infty$$
为了在视觉上形成连贯的底层射线，代码在渲染时计算其在屏幕底部以外的真实延伸落点（例：$y_{origin} = 1.15$），使得线条看起来真实地从画面外的脚底射入画面。

**2. 水平透视偏移量 ($x_{tip}$)**
基于 3D 世界里的转身角度 $\phi_{yaw}$，其在 2D 像素平面上的水平偏转落点为：
$$x_{tip} = 0.5 + 0.5 \cdot \frac{\tan(\phi_{yaw})}{\tan(\theta_h / 2)}$$

**3. 深度透视缩缩减与视觉探照射线 ($y_{tip}$)**
在 $H_c=0.6m$ 时，相机的可视地面最近距离为约 $0.97m$。如果严格按照 $0.3m$ 的物理真实步长去画线，整条轨迹将完全被压缩在画面最底部的画框黑边上，导致 VLM 无法通过画面感知纵深。
为此，代码引入了底线的向外发散 **视觉探射线**，长度强制拓展为： $D_{vis} = \max(2.5m, D_{step} \cdot 3)$。
随后使用余弦进行射线末端深度的 Z轴削减：
$$Z_{tip} = D_{vis} \cdot \cos(\phi_{yaw})$$
最后将深度投射反映到 y轴高度 上，越远越接近画面的中央 0.5 水平线：
$$y_{tip} = 0.5 + 0.5 \cdot \frac{(H_c / Z_{tip})}{\tan(\theta_v / 2)}$$

### 快速开始

运行端到端的 Habitat 导航 Agent：
```bash
conda run -n habitat python scripts/run_habitat_agent.py \
  --config configs/vlm_nav.json \
  --run-name my_first_run \
  --max-episodes 4 --max-steps 500
```
如果要纯离线地测试环境图以及地面光学投影画线效果，请运行 mini-dataset：
```bash
conda run -n habitat python scripts/run_minidataset.py \
  --config configs/vlm_nav.json \
  --limit 20 \
  --save-images
```
