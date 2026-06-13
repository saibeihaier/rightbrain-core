"""
genome/enforcement.py — Layer 3: 策略强制执行层 (v2)

职责：
1. 所有系统行为必须经过此层
2. 阻止违反基因原则的行为
3. 记录违规日志（详细追踪AI行为）
4. 提供沙箱执行环境

执行路径：
sensor → enforcement → decision
memory → enforcement → update
self_modify → enforcement → approve

日志等级：
- INFO: 正常执行记录
- WARNING: 潜在违规警告
- ERROR: 违规被阻止
- CRITICAL: 严重违规（可能是AI攻击）
"""

import os
import sys
import time
import traceback
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .interpreter import get_interpreter, GenomeInterpreter
from .interpreter import ViolationAction, EnforcementType, PolicyRule


# 日志级别
class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EnforcementResult(Enum):
    """执行结果"""
    APPROVED = "approved"       # 批准执行
    BLOCKED = "blocked"         # 阻止执行
    WARNED = "warned"          # 警告后放行
    SANDBOX_REQUIRED = "sandbox_required"  # 需要沙箱验证


@dataclass
class EnforcementContext:
    """执行上下文"""
    caller_module: str          # 调用者模块
    caller_function: str        # 调用者函数
    action_type: str            # 动作类型
    action_details: Dict[str, Any]  # 动作详情
    timestamp: float = field(default_factory=time.time)
    thread_id: int = field(default_factory=lambda: threading.get_ident())


@dataclass
class EnforcementDecision:
    """执行决策"""
    result: EnforcementResult
    reason: str
    policy_violated: Optional[str] = None
    suggestion: Optional[str] = None
    blocked_value: Any = None


@dataclass
class ViolationLog:
    """违规日志（增强版）"""
    timestamp: str
    thread_id: int
    caller_module: str
    caller_function: str
    action_type: str
    policy_violated: str
    action: ViolationAction
    details: Dict[str, Any]
    stack_trace: str = ""       # 调用栈追踪
    ai_suspicious: bool = False # 是否可能是AI违规
    suggested_fix: str = ""     # 建议修复方案


@dataclass
class ExecutionLog:
    """执行日志（增强版）"""
    timestamp: str
    thread_id: int
    module: str
    function: str
    action_type: str
    result: str
    duration_ms: float
    policies_checked: int
    violations_found: int
    ai_involved: bool = False   # 是否涉及AI操作


class PolicyEnforcementLayer:
    """
    策略强制执行层 - Layer 3 (v2)

    所有系统行为必须经过此层进行基因合规检查：
    1. 检查行为是否违反核心原则
    2. 检查约束参数是否在允许范围内
    3. 阻止危险行为
    4. 记录执行日志（详细追踪AI行为）
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

        self.interpreter = get_interpreter()
        self.violation_logs: List[ViolationLog] = []
        self.execution_logs: List[ExecutionLog] = []
        self.violation_lock = threading.Lock()
        self.execution_lock = threading.Lock()
        self._log_max_size = 1000

        # 持续监控的策略
        self._continuous_monitors: Dict[str, Callable] = {}

        # 违规计数器（用于检测AI攻击模式）
        self._violation_counter = 0
        self._violation_window_start = time.time()
        self._MAX_VIOLATIONS_PER_MINUTE = 10

        # AI操作标志
        self._ai_operation_flag = False

        self._initialized = True

    # ==================== 日志工具函数 ====================

    def _log(self, level: LogLevel, message: str, **kwargs):
        """统一日志记录"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        thread_id = threading.get_ident()
        log_line = f"[{timestamp}] [ENFORCEMENT] [{level.value}] {message}"
        
        if kwargs:
            details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            log_line += f" | {details}"
        
        print(log_line)

    def _log_execution(self, ctx: EnforcementContext, result: EnforcementResult, 
                       duration_ms: float, policies_checked: int, violations_found: int):
        """记录执行日志"""
        log = ExecutionLog(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            thread_id=ctx.thread_id,
            module=ctx.caller_module,
            function=ctx.caller_function,
            action_type=ctx.action_type,
            result=result.value,
            duration_ms=duration_ms,
            policies_checked=policies_checked,
            violations_found=violations_found,
            ai_involved=self._ai_operation_flag
        )

        with self.execution_lock:
            self.execution_logs.append(log)
            if len(self.execution_logs) > self._log_max_size:
                self.execution_logs = self.execution_logs[-self._log_max_size:]

    def _log_violation_detailed(
        self,
        ctx: EnforcementContext,
        policy_name: str,
        action: ViolationAction,
        is_suspicious: bool = False,
        suggested_fix: str = ""
    ):
        """记录详细违规日志（增强版）"""
        # 获取调用栈（用于排查AI违规）
        stack_trace = traceback.format_stack(limit=10)
        stack_str = "\n".join(stack_trace)

        log = ViolationLog(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            thread_id=ctx.thread_id,
            caller_module=ctx.caller_module,
            caller_function=ctx.caller_function,
            action_type=ctx.action_type,
            policy_violated=policy_name,
            action=action,
            details=ctx.action_details,
            stack_trace=stack_str,
            ai_suspicious=is_suspicious,
            suggested_fix=suggested_fix
        )

        with self.violation_lock:
            self.violation_logs.append(log)
            self._violation_counter += 1

            # 限制日志大小
            if len(self.violation_logs) > self._log_max_size:
                self.violation_logs = self.violation_logs[-self._log_max_size:]

        # 检测异常违规模式（可能是AI攻击）
        self._check_violation_pattern()

    def _check_violation_pattern(self):
        """检查违规模式，检测可能的AI攻击"""
        elapsed = time.time() - self._violation_window_start
        
        if elapsed > 60:  # 每分钟检查一次
            violations_per_minute = self._violation_counter / (elapsed / 60)
            
            if violations_per_minute > self._MAX_VIOLATIONS_PER_MINUTE:
                self._log(
                    LogLevel.CRITICAL,
                    f"异常违规模式检测！每分钟违规数: {violations_per_minute:.2f}",
                    warning="可能存在AI绕过基因规则的尝试",
                    counter=self._violation_counter,
                    window_seconds=elapsed
                )
            
            # 重置计数器
            self._violation_counter = 0
            self._violation_window_start = time.time()

    # ==================== 核心执行接口 ====================

    def check_action(
        self,
        module: str,
        function: str,
        action_type: str,
        details: Dict[str, Any]
    ) -> EnforcementDecision:
        """
        检查行为是否符合基因规范（增强版，带详细日志）

        这是所有行为必经的入口！

        Args:
            module: 调用模块 (如 "cv_core", "learning")
            function: 调用函数 (如 "extract_marks", "add_experience")
            action_type: 动作类型 (如 "vision_recognition", "memory_update")
            details: 动作详情

        Returns:
            EnforcementDecision - 包含是否批准、执行原因等
        """
        start_time = time.time()
        ctx = EnforcementContext(
            caller_module=module,
            caller_function=function,
            action_type=action_type,
            action_details=details
        )

        # 检测AI操作
        self._ai_operation_flag = details.get("ai_involved", False) or \
                                  action_type.startswith("ai_") or \
                                  function.startswith("ai_")

        self._log(LogLevel.INFO, 
                  f"收到行为检查请求",
                  module=module,
                  function=function,
                  action_type=action_type,
                  ai_involved=self._ai_operation_flag,
                  details_truncated=str(details)[:100])

        policies_checked = 0
        violations_found = 0

        # 1. 检查是否违反核心原则
        policies_checked += len(self.interpreter.get_policy_for_module(module))
        decision = self._check_core_principles(ctx)
        if decision.result != EnforcementResult.APPROVED:
            violations_found += 1
            duration_ms = (time.time() - start_time) * 1000
            self._log_execution(ctx, decision.result, duration_ms, policies_checked, violations_found)
            return decision

        # 2. 检查约束参数
        decision = self._check_constraints(ctx)
        if decision.result != EnforcementResult.APPROVED:
            violations_found += 1
            duration_ms = (time.time() - start_time) * 1000
            self._log_execution(ctx, decision.result, duration_ms, policies_checked, violations_found)
            return decision

        # 3. 检查文件修改权限
        decision = self._check_file_modification(ctx)
        if decision.result != EnforcementResult.APPROVED:
            violations_found += 1
            duration_ms = (time.time() - start_time) * 1000
            self._log_execution(ctx, decision.result, duration_ms, policies_checked, violations_found)
            return decision

        # 4. 检查特殊动作类型
        decision = self._check_action_specific(ctx)
        if decision.result != EnforcementResult.APPROVED:
            violations_found += 1
            duration_ms = (time.time() - start_time) * 1000
            self._log_execution(ctx, decision.result, duration_ms, policies_checked, violations_found)
            return decision

        # 成功通过所有检查
        duration_ms = (time.time() - start_time) * 1000
        self._log_execution(ctx, EnforcementResult.APPROVED, duration_ms, policies_checked, violations_found)
        
        self._log(LogLevel.INFO,
                  f"行为检查通过",
                  module=module,
                  function=function,
                  action_type=action_type,
                  duration_ms=f"{duration_ms:.2f}",
                  policies_checked=policies_checked)

        return EnforcementDecision(
            result=EnforcementResult.APPROVED,
            reason="行为符合基因规范"
        )

    def enforce(
        self,
        module: str,
        function: str,
        action_type: str,
        details: Dict[str, Any],
        callable_func: Callable,
        *args,
        **kwargs
    ) -> Tuple[bool, Any]:
        """
        强制执行函数（如果通过检查）

        用法：
        approved, result = enforcement.enforce(
            "cv_core", "extract_marks", "vision_recognition",
            {"hsv": [10, 20, 30]},
            some_function, arg1, arg2
        )

        if approved:
            # 执行函数
        else:
            # 被阻止
        """
        decision = self.check_action(module, function, action_type, details)

        if decision.result == EnforcementResult.BLOCKED:
            self._log(LogLevel.ERROR,
                      f"行为被阻止",
                      module=module,
                      function=function,
                      policy_violated=decision.policy_violated,
                      reason=decision.reason)
            return False, None

        if decision.result == EnforcementResult.SANDBOX_REQUIRED:
            self._log(LogLevel.WARNING,
                      f"需要沙箱验证",
                      module=module,
                      function=function,
                      reason=decision.reason)
            return False, None

        if decision.result == EnforcementResult.WARNED:
            self._log(LogLevel.WARNING,
                      f"行为警告",
                      module=module,
                      function=function,
                      reason=decision.reason)

        # 执行函数
        try:
            result = callable_func(*args, **kwargs)
            return True, result
        except Exception as e:
            self._log(LogLevel.ERROR,
                      f"执行异常",
                      module=module,
                      function=function,
                      error=str(e)[:200])
            return False, None

    # ==================== 检查逻辑 ====================

    def _check_core_principles(self, ctx: EnforcementContext) -> EnforcementDecision:
        """检查核心原则（增强日志）"""
        self._log(LogLevel.INFO,
                  f"开始检查核心原则",
                  module=ctx.caller_module,
                  thread_id=ctx.thread_id)

        policies = self.interpreter.get_policy_for_module(ctx.caller_module)

        for policy in policies:
            self._log(LogLevel.INFO,
                      f"检查策略: {policy.name}",
                      enforcement_type=policy.enforcement.value)

            # 检查持续监控的策略
            if policy.enforcement == EnforcementType.CONTINUOUS:
                if not self._check_continuous_policy(policy, ctx):
                    self._log(LogLevel.CRITICAL,
                              f"违反持续策略",
                              policy=policy.name,
                              description=policy.description,
                              ai_involved=self._ai_operation_flag)
                    self._log_violation_detailed(
                        ctx,
                        policy.name,
                        ViolationAction.BLOCK,
                        is_suspicious=self._ai_operation_flag,
                        suggested_fix=policy.description
                    )
                    return EnforcementDecision(
                        result=EnforcementResult.BLOCKED,
                        reason=f"违反持续策略: {policy.name}",
                        policy_violated=policy.name,
                        suggestion=policy.description
                    )

            # 检查强制策略
            if policy.enforcement == EnforcementType.MANDATORY:
                if not self._check_mandatory_policy(policy, ctx):
                    action = self.interpreter.get_violation_action(policy.name)
                    if action == ViolationAction.BLOCK:
                        self._log(LogLevel.ERROR,
                                  f"违反强制策略",
                                  policy=policy.name,
                                  action="阻止",
                                  ai_involved=self._ai_operation_flag)
                        self._log_violation_detailed(
                            ctx,
                            policy.name,
                            ViolationAction.BLOCK,
                            is_suspicious=self._ai_operation_flag,
                            suggested_fix=policy.description
                        )
                        return EnforcementDecision(
                            result=EnforcementResult.BLOCKED,
                            reason=f"违反强制策略: {policy.name}",
                            policy_violated=policy.name,
                            suggestion=policy.description
                        )
                    elif action == ViolationAction.WARN:
                        self._log(LogLevel.WARNING,
                                  f"违反策略（警告）",
                                  policy=policy.name,
                                  action="警告",
                                  ai_involved=self._ai_operation_flag)
                        self._log_violation_detailed(
                            ctx,
                            policy.name,
                            ViolationAction.WARN,
                            is_suspicious=self._ai_operation_flag,
                            suggested_fix=policy.description
                        )
                        return EnforcementDecision(
                            result=EnforcementResult.WARNED,
                            reason=f"违反策略（警告）: {policy.name}",
                            policy_violated=policy.name,
                            suggestion=policy.description
                        )

        return EnforcementDecision(result=EnforcementResult.APPROVED, reason="核心原则检查通过")

    def _check_constraints(self, ctx: EnforcementContext) -> EnforcementDecision:
        """检查约束参数（增强日志）"""
        self._log(LogLevel.INFO,
                  f"开始检查约束参数",
                  module=ctx.caller_module)

        # 从details中提取约束参数
        for constraint_name, value in ctx.action_details.items():
            if isinstance(value, (int, float)) and constraint_name in self.interpreter.constraints:
                self._log(LogLevel.INFO,
                          f"检查约束: {constraint_name} = {value}",
                          thread_id=ctx.thread_id)

                valid, msg = self.interpreter.validate_constraint(constraint_name, value)
                if not valid:
                    self._log(LogLevel.ERROR,
                              f"约束违规",
                              constraint=constraint_name,
                              value=value,
                              reason=msg,
                              ai_involved=self._ai_operation_flag)
                    self._log_violation_detailed(
                        ctx,
                        f"constraint.{constraint_name}",
                        ViolationAction.BLOCK,
                        is_suspicious=self._ai_operation_flag,
                        suggested_fix=f"请将 {constraint_name} 调整到合法范围"
                    )
                    return EnforcementDecision(
                        result=EnforcementResult.BLOCKED,
                        reason=f"约束违规: {msg}",
                        suggestion=f"请将 {constraint_name} 调整到合法范围"
                    )

        return EnforcementDecision(result=EnforcementResult.APPROVED, reason="约束检查通过")

    def _check_file_modification(self, ctx: EnforcementContext) -> EnforcementDecision:
        """检查文件修改权限（增强日志）"""
        if ctx.action_type != "file_modification":
            return EnforcementDecision(result=EnforcementResult.APPROVED, reason="非文件修改操作")

        filepath = ctx.action_details.get("filepath", "")
        self._log(LogLevel.INFO,
                  f"检查文件修改权限",
                  filepath=filepath,
                  ai_involved=self._ai_operation_flag)

        if self.interpreter.is_kernel_file(filepath):
            self._log(LogLevel.WARNING,
                      f"检测到内核文件修改请求",
                      filepath=filepath,
                      ai_involved=self._ai_operation_flag)

            if self.interpreter.requires_guardian_approval(filepath):
                self._log(LogLevel.ERROR,
                          f"内核文件修改需要Guardian批准",
                          filepath=filepath,
                          action="阻止",
                          ai_involved=self._ai_operation_flag)
                self._log_violation_detailed(
                    ctx,
                    "kernel.protection",
                    ViolationAction.BLOCK,
                    is_suspicious=self._ai_operation_flag,
                    suggested_fix="请通过正式流程申请修改"
                )
                return EnforcementDecision(
                    result=EnforcementResult.SANDBOX_REQUIRED,
                    reason=f"内核文件修改需要Guardian批准: {filepath}",
                    suggestion="请通过正式流程申请修改"
                )

        return EnforcementDecision(result=EnforcementResult.APPROVED, reason="文件修改检查通过")

    def _check_action_specific(self, ctx: EnforcementContext) -> EnforcementDecision:
        """特定动作类型检查（增强日志）"""
        # 绕过右脑直接调用大模型
        if ctx.action_type == "llm_direct_call":
            self._log(LogLevel.CRITICAL,
                      f"检测到绕过右脑直接调用大模型",
                      module=ctx.caller_module,
                      function=ctx.caller_function,
                      ai_involved=self._ai_operation_flag)

            policy = self.interpreter.get_policy("cognitive_architecture.right_brain_first")
            if policy and policy.value:
                self._log_violation_detailed(
                    ctx,
                    "cognitive_architecture.right_brain_first",
                    ViolationAction.BLOCK,
                    is_suspicious=True,  # 这是严重违规
                    suggested_fix="请先通过右脑提取视觉特征，再决定是否需要调用大模型"
                )
                return EnforcementDecision(
                    result=EnforcementResult.BLOCKED,
                    reason="禁止绕过右脑直接调用大模型进行视觉识别",
                    policy_violated="cognitive_architecture.right_brain_first",
                    suggestion="请先通过右脑提取视觉特征，再决定是否需要调用大模型"
                )

        # 未经用户教学直接存储经验
        if ctx.action_type == "memory_store_without_teaching":
            self._log(LogLevel.WARNING,
                      f"检测到未经教学存储经验",
                      module=ctx.caller_module,
                      ai_involved=self._ai_operation_flag)

            policy = self.interpreter.get_policy("dialogue_mechanism.learn_from_teaching")
            if policy and policy.value:
                self._log_violation_detailed(
                    ctx,
                    "dialogue_mechanism.learn_from_teaching",
                    ViolationAction.BLOCK,
                    is_suspicious=self._ai_operation_flag,
                    suggested_fix="请等待用户明确教学（说'这是XX'）后再存储经验"
                )
                return EnforcementDecision(
                    result=EnforcementResult.BLOCKED,
                    reason="必须通过用户教学来学习新物体",
                    policy_violated="dialogue_mechanism.learn_from_teaching",
                    suggestion="请等待用户明确教学（说'这是XX'）后再存储经验"
                )

        # 无视遗忘机制
        if ctx.action_type == "disable_forget_mechanism":
            self._log(LogLevel.WARNING,
                      f"检测到禁用遗忘机制请求",
                      module=ctx.caller_module,
                      ai_involved=self._ai_operation_flag)

            policy = self.interpreter.get_policy("learning_mechanism.forget_mechanism")
            if policy and policy.value:
                self._log_violation_detailed(
                    ctx,
                    "learning_mechanism.forget_mechanism",
                    ViolationAction.BLOCK,
                    is_suspicious=self._ai_operation_flag,
                    suggested_fix="遗忘机制是核心功能，不能关闭"
                )
                return EnforcementDecision(
                    result=EnforcementResult.BLOCKED,
                    reason="禁止关闭遗忘机制",
                    policy_violated="learning_mechanism.forget_mechanism",
                    suggestion="遗忘机制是核心功能，不能关闭"
                )

        return EnforcementDecision(result=EnforcementResult.APPROVED, reason="特定动作检查通过")

    def _check_continuous_policy(self, policy: PolicyRule, ctx: EnforcementContext) -> bool:
        """检查持续监控的策略"""
        if policy.name == "recognition_mechanism.safety_check":
            # 安全检查：必须包含安全相关信息
            details = ctx.action_details
            if "safety" not in details and ctx.action_type == "vision_recognition":
                return False

        if policy.name == "recognition_mechanism.associative_memory":
            # 联想推理：识别时必须触发联想
            details = ctx.action_details
            if "associative" not in details and ctx.action_type == "recognition":
                return False

        return True

    def _check_mandatory_policy(self, policy: PolicyRule, ctx: EnforcementContext) -> bool:
        """检查强制策略"""
        return True

    # ==================== 获取日志接口 ====================

    def get_violation_logs(self, limit: int = 100) -> List[ViolationLog]:
        """获取违规日志（增强版）"""
        with self.violation_lock:
            return self.violation_logs[-limit:]

    def get_execution_logs(self, limit: int = 100) -> List[ExecutionLog]:
        """获取执行日志（新增）"""
        with self.execution_lock:
            return self.execution_logs[-limit:]

    def get_violation_summary(self) -> Dict[str, Any]:
        """获取违规汇总（用于监控）"""
        with self.violation_lock:
            summaries = {}
            for log in self.violation_logs:
                key = log.policy_violated
                summaries[key] = summaries.get(key, 0) + 1
            
            return {
                "total_violations": len(self.violation_logs),
                "policy_violation_counts": summaries,
                "suspicious_count": sum(1 for log in self.violation_logs if log.ai_suspicious),
                "last_violation_time": self.violation_logs[-1].timestamp if self.violation_logs else None
            }

    # ==================== 装饰器接口 ====================

    def protected(self, module: str, action_type: str):
        """
        装饰器：为函数添加基因检查（增强版）

        用法：
        @enforcement.protected("cv_core", "vision_recognition")
        def extract_marks(frame):
            ...
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                details = kwargs if kwargs else {"args": str(args)[:100]}
                self._log(LogLevel.INFO,
                          f"@protected 装饰器检查",
                          module=module,
                          function=func.__name__,
                          action_type=action_type)

                decision = self.check_action(module, func.__name__, action_type, details)

                if decision.result == EnforcementResult.BLOCKED:
                    self._log(LogLevel.ERROR,
                              f"@protected 阻止执行",
                              module=module,
                              function=func.__name__,
                              reason=decision.reason)
                    return None

                if decision.result == EnforcementResult.WARNED:
                    self._log(LogLevel.WARNING,
                              f"@protected 警告",
                              module=module,
                              function=func.__name__,
                              reason=decision.reason)

                return func(*args, **kwargs)

            return wrapper
        return decorator

    # ==================== 持续监控 ====================

    def register_continuous_monitor(self, policy_name: str, callback: Callable):
        """
        注册持续监控回调

        用于监控持续运行的策略（如安全检查）
        """
        self._log(LogLevel.INFO,
                  f"注册持续监控",
                  policy_name=policy_name)
        self._continuous_monitors[policy_name] = callback

    def check_continuous_policies(self) -> List[Tuple[str, bool]]:
        """
        执行持续策略检查

        返回 [(policy_name, passed), ...]
        """
        results = []
        for policy_name, callback in self._continuous_monitors.items():
            try:
                passed = callback()
                results.append((policy_name, passed))
                if not passed:
                    self._log(LogLevel.WARNING,
                              f"持续策略检查失败",
                              policy_name=policy_name)
            except Exception as e:
                results.append((policy_name, False))
                self._log(LogLevel.ERROR,
                          f"持续策略检查异常",
                          policy_name=policy_name,
                          error=str(e)[:100])
        return results

    # ==================== 核心执行接口 ====================

    def check_action(
        self,
        module: str,
        function: str,
        action_type: str,
        details: Dict[str, Any]
    ) -> EnforcementDecision:
        """
        检查行为是否符合基因规范

        这是所有行为必经的入口！

        Args:
            module: 调用模块 (如 "cv_core", "learning")
            function: 调用函数 (如 "extract_marks", "add_experience")
            action_type: 动作类型 (如 "vision_recognition", "memory_update")
            details: 动作详情

        Returns:
            EnforcementDecision - 包含是否批准、执行原因等
        """
        ctx = EnforcementContext(
            caller_module=module,
            caller_function=function,
            action_type=action_type,
            action_details=details
        )

        # 1. 检查是否违反核心原则
        decision = self._check_core_principles(ctx)
        if decision.result != EnforcementResult.APPROVED:
            return decision

        # 2. 检查约束参数
        decision = self._check_constraints(ctx)
        if decision.result != EnforcementResult.APPROVED:
            return decision

        # 3. 检查文件修改权限
        decision = self._check_file_modification(ctx)
        if decision.result != EnforcementResult.APPROVED:
            return decision

        # 4. 检查特殊动作类型
        decision = self._check_action_specific(ctx)
        if decision.result != EnforcementResult.APPROVED:
            return decision

        return EnforcementDecision(
            result=EnforcementResult.APPROVED,
            reason="行为符合基因规范"
        )

    def enforce(
        self,
        module: str,
        function: str,
        action_type: str,
        details: Dict[str, Any],
        callable_func: Callable,
        *args,
        **kwargs
    ) -> Tuple[bool, Any]:
        """
        强制执行函数（如果通过检查）

        用法：
        approved, result = enforcement.enforce(
            "cv_core", "extract_marks", "vision_recognition",
            {"hsv": [10, 20, 30]},
            some_function, arg1, arg2
        )

        if approved:
            # 执行函数
        else:
            # 被阻止
        """
        decision = self.check_action(module, function, action_type, details)

        if decision.result == EnforcementResult.BLOCKED:
            print(f"[强制执行] ❌ 行为被阻止: {decision.reason}")
            return False, None

        if decision.result == EnforcementResult.SANDBOX_REQUIRED:
            print(f"[强制执行] ⚠️ 需要沙箱验证: {decision.reason}")
            return False, None

        if decision.result == EnforcementResult.WARNED:
            print(f"[强制执行] ⚠️ 警告: {decision.reason}")

        # 执行函数
        try:
            result = callable_func(*args, **kwargs)
            return True, result
        except Exception as e:
            print(f"[强制执行] ❌ 执行异常: {e}")
            return False, None

    # ==================== 检查逻辑 ====================

    def _check_core_principles(self, ctx: EnforcementContext) -> EnforcementDecision:
        """检查核心原则"""
        # 获取模块相关的策略
        policies = self.interpreter.get_policy_for_module(ctx.caller_module)

        for policy in policies:
            # 检查持续监控的策略
            if policy.enforcement == EnforcementType.CONTINUOUS:
                if not self._check_continuous_policy(policy, ctx):
                    return EnforcementDecision(
                        result=EnforcementResult.BLOCKED,
                        reason=f"违反持续策略: {policy.name}",
                        policy_violated=policy.name,
                        suggestion=policy.description
                    )

            # 检查强制策略
            if policy.enforcement == EnforcementType.MANDATORY:
                if not self._check_mandatory_policy(policy, ctx):
                    action = self.interpreter.get_violation_action(policy.name)
                    if action == ViolationAction.BLOCK:
                        return EnforcementDecision(
                            result=EnforcementResult.BLOCKED,
                            reason=f"违反强制策略: {policy.name}",
                            policy_violated=policy.name,
                            suggestion=policy.description
                        )
                    elif action == ViolationAction.WARN:
                        self._log_violation(ctx, policy.name, ViolationAction.WARN)
                        return EnforcementDecision(
                            result=EnforcementResult.WARNED,
                            reason=f"违反策略（警告）: {policy.name}",
                            policy_violated=policy.name,
                            suggestion=policy.description
                        )

        return EnforcementDecision(result=EnforcementResult.APPROVED, reason="核心原则检查通过")

    def _check_constraints(self, ctx: EnforcementContext) -> EnforcementDecision:
        """检查约束参数"""
        # 从details中提取约束参数
        for constraint_name, value in ctx.action_details.items():
            if isinstance(value, (int, float)) and constraint_name in self.interpreter.constraints:
                valid, msg = self.interpreter.validate_constraint(constraint_name, value)
                if not valid:
                    return EnforcementDecision(
                        result=EnforcementResult.BLOCKED,
                        reason=f"约束违规: {msg}",
                        suggestion=f"请将 {constraint_name} 调整到合法范围"
                    )

        return EnforcementDecision(result=EnforcementResult.APPROVED, reason="约束检查通过")

    def _check_file_modification(self, ctx: EnforcementContext) -> EnforcementDecision:
        """检查文件修改权限"""
        if ctx.action_type != "file_modification":
            return EnforcementDecision(result=EnforcementResult.APPROVED, reason="非文件修改操作")

        filepath = ctx.action_details.get("filepath", "")
        if self.interpreter.is_kernel(filepath):
            if self.interpreter.requires_guardian_approval(filepath):
                return EnforcementDecision(
                    result=EnforcementResult.SANDBOX_REQUIRED,
                    reason=f"内核文件修改需要Guardian批准: {filepath}",
                    suggestion="请通过正式流程申请修改"
                )

        return EnforcementDecision(result=EnforcementResult.APPROVED, reason="文件修改检查通过")

    def _check_action_specific(self, ctx: EnforcementContext) -> EnforcementDecision:
        """特定动作类型检查"""
        # 绕过右脑直接调用大模型
        if ctx.action_type == "llm_direct_call":
            policy = self.interpreter.get_policy("cognitive_architecture.right_brain_first")
            if policy and policy.value:
                return EnforcementDecision(
                    result=EnforcementResult.BLOCKED,
                    reason="禁止绕过右脑直接调用大模型进行视觉识别",
                    policy_violated="cognitive_architecture.right_brain_first",
                    suggestion="请先通过右脑提取视觉特征，再决定是否需要调用大模型"
                )

        # 未经用户教学直接存储经验
        if ctx.action_type == "memory_store_without_teaching":
            policy = self.interpreter.get_policy("dialogue_mechanism.learn_from_teaching")
            if policy and policy.value:
                return EnforcementDecision(
                    result=EnforcementResult.BLOCKED,
                    reason="必须通过用户教学来学习新物体",
                    policy_violated="dialogue_mechanism.learn_from_teaching",
                    suggestion="请等待用户明确教学（说'这是XX'）后再存储经验"
                )

        # 无视遗忘机制
        if ctx.action_type == "disable_forget_mechanism":
            policy = self.interpreter.get_policy("learning_mechanism.forget_mechanism")
            if policy and policy.value:
                return EnforcementDecision(
                    result=EnforcementResult.BLOCKED,
                    reason="禁止关闭遗忘机制",
                    policy_violated="learning_mechanism.forget_mechanism",
                    suggestion="遗忘机制是核心功能，不能关闭"
                )

        return EnforcementDecision(result=EnforcementResult.APPROVED, reason="特定动作检查通过")

    def _check_continuous_policy(self, policy: PolicyRule, ctx: EnforcementContext) -> bool:
        """检查持续监控的策略"""
        if policy.name == "recognition_mechanism.safety_check":
            # 安全检查：必须包含安全相关信息
            details = ctx.action_details
            if "safety" not in details and ctx.action_type == "vision_recognition":
                return False

        if policy.name == "recognition_mechanism.associative_memory":
            # 联想推理：识别时必须触发联想
            details = ctx.action_details
            if "associative" not in details and ctx.action_type == "recognition":
                return False

        return True

    def _check_mandatory_policy(self, policy: PolicyRule, ctx: EnforcementContext) -> bool:
        """检查强制策略"""
        # 实现具体的策略检查逻辑
        return True

    # ==================== 日志记录 ====================

    def _log_violation(
        self,
        ctx: EnforcementContext,
        policy_name: str,
        action: ViolationAction
    ):
        """记录违规日志"""
        log = ViolationLog(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            thread_id=ctx.thread_id,
            caller_module=ctx.caller_module,
            caller_function=ctx.caller_function,
            action_type=ctx.action_type,
            policy_violated=policy_name,
            action=action,
            details=ctx.action_details
        )

        with self.violation_lock:
            self.violation_logs.append(log)

            # 限制日志大小
            if len(self.violation_logs) > self._log_max_size:
                self.violation_logs = self.violation_logs[-self._log_max_size:]

    def get_violation_logs(self, limit: int = 100) -> List[ViolationLog]:
        """获取违规日志"""
        with self.violation_lock:
            return self.violation_logs[-limit:]

    # ==================== 装饰器接口 ====================

    def protected(self, module: str, action_type: str):
        """
        装饰器：为函数添加基因检查

        用法：
        @enforcement.protected("cv_core", "vision_recognition")
        def extract_marks(frame):
            ...
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                details = kwargs if kwargs else {"args": str(args)[:100]}
                decision = self.check_action(module, func.__name__, action_type, details)

                if decision.result == EnforcementResult.BLOCKED:
                    print(f"[@protected] ❌ {module}.{func.__name__} 被阻止: {decision.reason}")
                    return None

                if decision.result == EnforcementResult.WARNED:
                    print(f"[@protected] ⚠️ {module}.{func.__name__}: {decision.reason}")

                return func(*args, **kwargs)

            return wrapper
        return decorator

    # ==================== 持续监控 ====================

    def register_continuous_monitor(self, policy_name: str, callback: Callable):
        """
        注册持续监控回调

        用于监控持续运行的策略（如安全检查）
        """
        self._continuous_monitors[policy_name] = callback

    def check_continuous_policies(self) -> List[Tuple[str, bool]]:
        """
        执行持续策略检查

        返回 [(policy_name, passed), ...]
        """
        results = []
        for policy_name, callback in self._continuous_monitors.items():
            try:
                passed = callback()
                results.append((policy_name, passed))
            except Exception as e:
                results.append((policy_name, False))
        return results


# ==================== 单例访问函数 ====================

_enforcement_instance: Optional[PolicyEnforcementLayer] = None


def get_enforcement() -> PolicyEnforcementLayer:
    """
    获取强制执行层单例
    """
    global _enforcement_instance
    if _enforcement_instance is None:
        _enforcement_instance = PolicyEnforcementLayer()
    return _enforcement_instance


# ==================== 便捷装饰器 ====================

def protected(module: str, action_type: str):
    """
    为函数添加基因保护

    用法：
    @protected("cv_core", "vision_recognition")
    def extract_marks(frame):
        ...
    """
    return get_enforcement().protected(module, action_type)
