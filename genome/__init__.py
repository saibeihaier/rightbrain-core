"""
RightBrain 基因定义 v2

这是项目的 DNA 入口点。四层架构：

Layer 1: genome_spec.json — 纯声明层（JSON）
Layer 2: interpreter.py — 基因解释器
Layer 3: enforcement.py — 强制执行层
Layer 4: reflection.py — 自检层

使用方式：

    # 校验基因完整性
    from genome import check_genome_integrity

    # AI获取只读视图（推荐方式）
    from genome import get_ai_view
    view = get_ai_view()

    # 获取特定策略
    from genome import get_interpreter
    policy = get_interpreter().get_policy("cognitive_architecture.right_brain_first")

    # AI自我检查
    from genome import ai_self_check
    result = ai_self_check("cv_core", "extract_marks", {"action": "vision"})

    # 提出进化申请
    from genome import propose_evolution, ModificationType
    request_id = propose_evolution(
        ModificationType.CONSTRAINT_TUNING,
        "调整置信度阈值",
        "confidence_threshold",
        0.6,
        0.7,
        "提高识别精度"
    )
"""

# v2 核心导出
from .genome import (
    GENOME_VERSION,
    check_genome_integrity,
    get_genome_hash,
    get_version,
    get_interpreter,
    get_enforcement,
    get_reflection,
    get_ai_view,
)

# 解释器导出
from .interpreter import (
    GenomeInterpreter,
    get_policy,
    get_constraint,
    validate_value,
    is_kernel,
    PolicyRule,
    ConstraintParam,
)

# 强制执行层导出
from .enforcement import (
    PolicyEnforcementLayer,
    EnforcementResult,
    EnforcementDecision,
    EnforcementContext,
    protected,
)

# 自检层导出
from .reflection import (
    ReflectionLayer,
    SelfCheckResult,
    EvolutionRequest,
    ModificationType,
    EvolutionRequestStatus,
    ai_self_check,
    propose_evolution,
)

# 向后兼容：旧版本的核心算法函数
# 注意：这些函数现在仅供参考，实际逻辑在genome_algorithms.py中
from .genome_algorithms import (
    color_detection_genome,
    shape_correction_genome,
    shape_long_rect_genome,
    wta_decision_genome,
    memory_match_genome,
)

# 向后兼容：直接从spec加载CORE_PRINCIPLES和CORE_CONSTRAINTS
# 通过genome.genome模块的函数访问
from .genome import get_core_principles, get_core_constraints

# 导出为模块级常量（兼容v1的字典访问）
CORE_PRINCIPLES = get_core_principles()
CORE_CONSTRAINTS = get_core_constraints()
