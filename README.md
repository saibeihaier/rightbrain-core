# RightBrain Core — 给大模型的右脑

> RightBrain Core 是认知引擎的核心库，不包含 GUI、语音识别或摄像头驱动。
> 完整的桌面应用（含 GUI、语音对话、130条种子经验）在作者私有仓库。

RightBrain 不是一个 AI 模型。它是一个独立的感知系统——有眼睛、有直觉、有记忆——专门做"大模型不擅长的事"：看清世界、记住事物、判断危险、联想推理。大模型负责说话和思考，RightBrain 负责看和感受。

## 核心理念

所有 AI 都是"左脑"——处理语言、逻辑、符号，但看不到世界。给 GPT 一张图，它"看"到的是像素矩阵，不是"苹果在桌子边缘要掉了"。

RightBrain 是**右脑**。它不处理语言，它处理视觉、空间、直觉、经验：

- 看到一个物体 → 自己认出来（不用问服务器）
- 认不出来 → 在已有经验里联想（红圆形→可能是苹果）
- 物体在桌子边缘 → 直觉上感觉"危险"
- 你教一次 → 永远记住

### 左右脑分工

| 左脑（大模型） | 右脑（RightBrain） |
|--------------|------------------|
| 理解语言 | 看世界 |
| 逻辑推理 | 直觉判断 |
| 知识问答 | 经验匹配 |
| 对话交流 | 空间感知 |

### 基因设计

RightBrain 的核心原则是锁死的——这叫**基因**。任何人（包括 AI 工具）修改代码时，都不能删除"右脑必须先处理视觉信息"、不能关闭"安全检查"、不能绕过"经验学习"。

## 快速开始

```bash
pip install -r requirements.txt

# Python 中直接使用
from rightbrain.memory2 import ExperienceMemory2
from rightbrain.sensor2 import extract_marks
from rightbrain.decision2 import identify_v2

# 加载经验库
mem = ExperienceMemory2("data/seed_50.json")

# 识别物体
marks = extract_marks(image, bbox, full_analysis=True)
action, score, exp, _ = identify_v2(marks, mem)
print(f"识别结果: {action} ({score:.2f})")
```

## 系统架构

```
输入图像
    ↓
sensor2 — 视觉特征提取（颜色、形状、纹理、空间）
    ↓
memory2 — WTA竞争经验匹配
    ↓
decision2 — 识别输出 + 安全检查
    ↓
输出JSON场景报告（任何大模型都能用）

稳定性层（确保系统不漂移）：
  boundary_enforcer — 参数受基因范围钳制
  drift_monitor — 慢性漂移检测
  adaptive_stabilizer — 自动回到好状态

基因（不可变核心）：
  genome_spec.json — 核心原则 + 参数范围
  SHA256校验 — 防篡改
```

## 核心模块

| 模块 | 文件 | 功能 |
|------|------|------|
| 感知器 | `sensor2.py` | 颜色/形状/纹理/空间/安全检查 |
| 经验库 | `memory2.py` | 稀疏标记存储、WTA竞争匹配 |
| 决策 | `decision2.py` | 识别输出、安全检查 |
| 联想推理 | `associative_memory.py` | 类比推理、链式联想、形象思维 |
| 对话 | `dialogue.py` | 状态机、用户教学、技能调用 |
| 基因 | `genome/` | 核心原则、算法基因、SHA256校验 |
| 稳定层 | `stability.py` | 探针采样、漂移检测、自适应纠正 |
| 边界执行 | `boundary_enforcer.py` | 参数钳制、好状态锚点、漂移自动回调 |

## 许可证

MIT License
