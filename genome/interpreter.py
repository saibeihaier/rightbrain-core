"""
genome/interpreter.py — Layer 2: 基因解释器

职责：
1. 将 genome_spec.json 解析为运行时策略
2. 提供只读的"策略视图"给AI
3. 不执行任何修改，只解释规则

AI只能通过此模块读取基因策略，不能直接修改genome_spec.json
"""

import json
import os
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ImmutabilityLevel(Enum):
    """不可变性级别"""
    KERNEL = "kernel"      # 内核文件，完全不可改
    CONFIG = "config"      # 配置文件，可通过批准修改
    DATA = "data"          # 数据文件，可正常修改
    RUNTIME = "runtime"    # 运行时生成，无需保护


class ViolationAction(Enum):
    """违规处理方式"""
    BLOCK = "block"        # 立即阻止
    WARN = "warn"          # 警告但允许
    IGNORE = "ignore"      # 忽略


class EnforcementType(Enum):
    """执行类型"""
    MANDATORY = "mandatory"    # 强制执行
    CONTINUOUS = "continuous"   # 持续监控
    OPTIONAL = "optional"       # 可选


@dataclass
class PolicyRule:
    """单条策略规则"""
    name: str
    value: Any
    description: str
    enforcement: EnforcementType
    violation_action: ViolationAction
    ai_readonly: bool = True


@dataclass
class ConstraintParam:
    """约束参数"""
    name: str
    min_val: float
    max_val: float
    default: float
    unit: str


@dataclass
class AlgorithmSpec:
    """算法规范"""
    name: str
    type: str
    description: str
    rules: List[Dict]
    ai_readonly: bool = True


class GenomeInterpreter:
    """
    基因解释器 - Layer 2

    职责：
    1. 加载和解析 genome_spec.json
    2. 将声明式规则转换为运行时策略
    3. 提供只读的策略查询接口
    4. 验证约束参数是否在允许范围内
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.spec_path = os.path.join(
            os.path.dirname(__file__),
            "genome_spec.json"
        )
        self.spec: Dict[str, Any] = {}
        self.policies: Dict[str, PolicyRule] = {}
        self.constraints: Dict[str, ConstraintParam] = {}
        self.algorithm_specs: Dict[str, AlgorithmSpec] = {}
        self.kernel_files: List[str] = []
        self.hash: str = ""

        self._load_spec()
        self._parse_policies()
        self._parse_constraints()
        self._parse_algorithms()
        self._parse_kernel_files()

        self._initialized = True

    def _load_spec(self):
        """加载基因组声明文件"""
        with open(self.spec_path, 'r', encoding='utf-8') as f:
            self.spec = json.load(f)

        # 计算spec的哈希（用于检测篡改）
        self.hash = self._compute_spec_hash()

    def _compute_spec_hash(self) -> str:
        """计算spec文件的哈希"""
        with open(self.spec_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _parse_policies(self):
        """解析核心策略"""
        principles = self.spec.get("core_principles", {})

        for category, items in principles.items():
            for name, data in items.items():
                self.policies[f"{category}.{name}"] = PolicyRule(
                    name=f"{category}.{name}",
                    value=data.get("value"),
                    description=data.get("desc", ""),
                    enforcement=EnforcementType(data.get("enforcement", "mandatory")),
                    violation_action=ViolationAction(data.get("violation_action", "block")),
                    ai_readonly=True
                )

    def _parse_constraints(self):
        """解析约束参数"""
        constraints = self.spec.get("core_constraints", {})

        for name, data in constraints.items():
            if name.startswith("_"):
                continue

            self.constraints[name] = ConstraintParam(
                name=name,
                min_val=data.get("min", 0),
                max_val=data.get("max", 1),
                default=data.get("default", 0.5),
                unit=data.get("unit", "ratio")
            )

    def _parse_algorithms(self):
        """解析算法规范"""
        algos = self.spec.get("algorithm_specs", {})

        for name, data in algos.items():
            if name.startswith("_"):
                continue

            self.algorithm_specs[name] = AlgorithmSpec(
                name=name,
                type=data.get("type", ""),
                description=data.get("description", ""),
                rules=data.get("rules", []),
                ai_readonly=data.get("ai_readonly", True)
            )

    def _parse_kernel_files(self):
        """解析内核文件列表"""
        kernel = self.spec.get("kernel_files", {})
        self.kernel_files = kernel.get("files", [])

    # ==================== 公开接口（AI可调用） ====================

    def get_policy(self, policy_name: str) -> Optional[PolicyRule]:
        """
        获取策略规则（只读视图）

        AI调用方式：
        policy = genome_engine.get_policy("cognitive_architecture.right_brain_first")

        返回：
        - PolicyRule 对象（包含 value, description, enforcement 等）
        - None（如果策略不存在）
        """
        return self.policies.get(policy_name)

    def get_all_policies(self) -> Dict[str, PolicyRule]:
        """
        获取所有策略（只读视图）

        AI可调用此方法查看所有可用策略
        """
        return self.policies.copy()

    def get_constraint(self, constraint_name: str) -> Optional[ConstraintParam]:
        """
        获取约束参数范围

        AI调用方式：
        constraint = genome_engine.get_constraint("confidence_threshold")
        if 0.1 <= value <= constraint.max_val:
            # 值合法
        """
        return self.constraints.get(constraint_name)

    def validate_constraint(self, name: str, value: float) -> Tuple[bool, str]:
        """
        验证约束值是否在允许范围内

        返回：(是否合法, 错误信息)
        """
        constraint = self.constraints.get(name)
        if constraint is None:
            return False, f"未知约束: {name}"

        if not (constraint.min_val <= value <= constraint.max_val):
            return False, (
                f"约束 {name} 的值 {value} 超出范围 "
                f"[{constraint.min_val}, {constraint.max_val}]"
            )

        return True, "合法"

    def get_algorithm_spec(self, algorithm_name: str) -> Optional[AlgorithmSpec]:
        """
        获取算法规范（只读视图）

        AI可调用此方法了解算法规则，但不能修改
        """
        return self.algorithm_specs.get(algorithm_name)

    def is_kernel_file(self, filepath: str) -> bool:
        """
        检查文件是否为核心内核文件

        内核文件修改需要Guardian批准
        """
        basename = os.path.basename(filepath)
        return basename in [os.path.basename(f) for f in self.kernel_files]

    def requires_guardian_approval(self, filepath: str) -> bool:
        """
        检查文件修改是否需要Guardian批准
        """
        if not self.is_kernel_file(filepath):
            return False
        return self.spec.get("kernel_files", {}).get("require_guardian_approval", True)

    def get_violation_action(self, policy_name: str) -> ViolationAction:
        """
        获取策略违规时的处理方式
        """
        policy = self.policies.get(policy_name)
        if policy:
            return policy.violation_action
        return ViolationAction.BLOCK  # 默认阻止

    def get_enforcement_type(self, policy_name: str) -> EnforcementType:
        """
        获取策略的执行类型
        """
        policy = self.policies.get(policy_name)
        if policy:
            return policy.enforcement
        return EnforcementType.MANDATORY  # 默认强制

    def get_policy_for_module(self, module_name: str) -> List[PolicyRule]:
        """
        获取指定模块需要遵守的策略

        Args:
            module_name: 模块名，如 "cv_core", "learning", "dialogue"
        """
        module_policies = []

        for name, policy in self.policies.items():
            # 策略名格式：category.name
            # 例如：cognitive_architecture.right_brain_first
            if module_name in name:
                module_policies.append(policy)

        return module_policies

    def get_evolution_policy(self) -> Dict[str, Any]:
        """
        获取进化策略配置
        """
        return self.spec.get("evolution_policy", {})

    def get_spec_hash(self) -> str:
        """
        获取spec文件的哈希（用于检测篡改）
        """
        return self.hash

    def get_version(self) -> str:
        """
        获取基因组版本
        """
        return self.spec.get("genome_info", {}).get("version", "1.0.0")

    # ==================== AI只读视图 ====================

    def get_ai_readonly_view(self) -> Dict[str, Any]:
        """
        获取AI只读视图

        AI只能通过此方法获取基因信息，不能直接访问spec
        """
        view = {
            "version": self.get_version(),
            "policies": {},
            "constraints": {},
            "algorithms": {},
            "kernel_files": self.kernel_files,
            "immutability_notice": (
                "这些规则是不可变的。修改需要通过正式的进化流程。"
            )
        }

        for name, policy in self.policies.items():
            view["policies"][name] = {
                "value": policy.value,
                "description": policy.description,
                "enforcement": policy.enforcement.value,
                "violation_action": policy.violation_action.value,
                "readonly": True
            }

        for name, constraint in self.constraints.items():
            view["constraints"][name] = {
                "min": constraint.min_val,
                "max": constraint.max_val,
                "default": constraint.default,
                "unit": constraint.unit
            }

        for name, algo in self.algorithm_specs.items():
            view["algorithms"][name] = {
                "type": algo.type,
                "description": algo.description,
                "readonly": algo.ai_readonly
            }

        return view


# ==================== 单例访问函数 ====================

_interpreter_instance: Optional[GenomeInterpreter] = None


def get_interpreter() -> GenomeInterpreter:
    """
    获取基因解释器单例

    这是AI读取基因策略的唯一入口
    """
    global _interpreter_instance
    if _interpreter_instance is None:
        _interpreter_instance = GenomeInterpreter()
    return _interpreter_instance


# ==================== 便捷函数（供其他模块使用） ====================

def get_policy(name: str) -> Optional[PolicyRule]:
    """获取单条策略"""
    return get_interpreter().get_policy(name)


def get_constraint(name: str) -> Optional[ConstraintParam]:
    """获取约束参数"""
    return get_interpreter().get_constraint(name)


def validate_value(name: str, value: float) -> Tuple[bool, str]:
    """验证约束值"""
    return get_interpreter().validate_constraint(name, value)


def is_kernel(filepath: str) -> bool:
    """检查是否为核心文件"""
    return get_interpreter().is_kernel_file(filepath)


def get_ai_view() -> Dict[str, Any]:
    """获取AI只读视图"""
    return get_interpreter().get_ai_readonly_view()
