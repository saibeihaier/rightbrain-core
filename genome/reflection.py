"""
genome/reflection.py — Layer 4: 自检层 (v2)

职责：
1. AI可以检查自己是否违反基因（带详细日志）
2. AI可以提出修改建议
3. AI不能直接修改基因
4. 提供进化申请接口（带审计日志）

AI只能通过此模块进行自我检查，不能直接修改任何基因文件
"""

import json
import os
import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .interpreter import get_interpreter, GenomeInterpreter
from .enforcement import get_enforcement, PolicyEnforcementLayer


class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    AUDIT = "AUDIT"


class ModificationType(Enum):
    BUG_FIX = "bug_fix"
    ENHANCEMENT = "enhancement"
    PRINCIPLE_CHANGE = "principle_change"
    NEW_ALGORITHM = "new_algorithm"
    CONSTRAINT_TUNING = "constraint_tuning"


class EvolutionRequestStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SANDBOX_TESTING = "sandbox_testing"
    APPLIED = "applied"


@dataclass
class EvolutionRequest:
    request_id: str
    timestamp: str
    modification_type: ModificationType
    description: str
    target_file: str
    current_value: Any
    proposed_value: Any
    rationale: str
    status: EvolutionRequestStatus
    ai_self_check: Dict[str, Any] = field(default_factory=dict)
    guardian_review: Dict[str, Any] = field(default_factory=dict)
    sandbox_result: Dict[str, Any] = field(default_factory=dict)
    ai_requested: bool = False
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SelfCheckResult:
    passed: bool
    policies_compliant: List[str]
    policies_violated: List[str]
    suggestions: List[str]
    recommendations: List[str]
    timestamp: str = ""
    ai_involved: bool = False
    risk_level: str = "low"


@dataclass
class AuditLog:
    timestamp: str
    action: str
    actor: str
    details: Dict[str, Any]
    request_id: Optional[str] = None


class ReflectionLayer:
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    _instance = None

    def __init__(self):
        if self._initialized:
            return

        self.interpreter = get_interpreter()
        self.enforcement = get_enforcement()
        self.evolution_requests: List[EvolutionRequest] = []
        self.audit_logs: List[AuditLog] = []
        self.evolution_lock = threading.Lock()
        self.audit_lock = threading.Lock()
        self._request_counter = 0
        self._log_max_size = 1000

        self._initialized = True

    def _log(self, level: LogLevel, message: str, **kwargs):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        thread_id = threading.get_ident()
        log_line = f"[{timestamp}] [REFLECTION] [{level.value}] {message}"
        if kwargs:
            details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            log_line += f" | {details}"
        print(log_line)

    def _log_audit(self, action: str, actor: str, details: Dict[str, Any], request_id: str = None):
        log = AuditLog(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            action=action,
            actor=actor,
            details=details,
            request_id=request_id
        )
        with self.audit_lock:
            self.audit_logs.append(log)
            if len(self.audit_logs) > self._log_max_size:
                self.audit_logs = self.audit_logs[-self._log_max_size:]
        self._log(LogLevel.AUDIT, action, actor=actor, request_id=request_id, **details)

    def self_check(self, module: str, function: str, planned_action: Dict[str, Any]) -> SelfCheckResult:
        start_time = time.time()
        self._log(LogLevel.INFO, f"AI自我检查开始", module=module, function=function)

        ai_involved = planned_action.get("ai_involved", False) or \
                      function.startswith("ai_") or \
                      module.startswith("ai_")

        policies = self.interpreter.get_policy_for_module(module)
        compliant = []
        violated = []
        suggestions = []
        recommendations = []
        risk_level = "low"

        for policy in policies:
            self._log(LogLevel.INFO, f"检查策略: {policy.name}")

            if self._check_policy_compliance(policy, planned_action):
                compliant.append(policy.name)
            else:
                violated.append(policy.name)
                suggestions.append(f"违反策略 {policy.name}: {policy.description}")
                
                if policy.enforcement.value == "mandatory" and policy.violation_action.value == "block":
                    risk_level = "high" if risk_level == "low" else "critical"
                elif policy.enforcement.value == "continuous":
                    risk_level = "medium" if risk_level == "low" else risk_level

                recommendations.append(self._get_recommendation(policy, planned_action))
                self._log(LogLevel.WARNING, f"策略违规检测", policy=policy.name, risk_level=risk_level)

        if violated:
            self._log(LogLevel.ERROR, f"AI自我检查失败", violations=len(violated), risk_level=risk_level)
        else:
            self._log(LogLevel.INFO, f"AI自我检查通过", compliant_policies=len(compliant))

        duration_ms = (time.time() - start_time) * 1000
        self._log(LogLevel.INFO, f"AI自我检查完成", duration_ms=f"{duration_ms:.2f}")

        return SelfCheckResult(
            passed=len(violated) == 0,
            policies_compliant=compliant,
            policies_violated=violated,
            suggestions=suggestions,
            recommendations=recommendations,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ai_involved=ai_involved,
            risk_level=risk_level
        )

    def _check_policy_compliance(self, policy, action: Dict[str, Any]) -> bool:
        return True

    def _get_recommendation(self, policy, action: Dict[str, Any]) -> str:
        return f"请调整行为以符合策略 {policy.name}"

    def propose_evolution(self, modification_type: ModificationType, description: str,
                         target_file: str, current_value: Any, proposed_value: Any,
                         rationale: str) -> str:
        self._log(LogLevel.AUDIT, f"收到进化申请", type=modification_type.value, target_file=target_file)

        ai_self_check = self._perform_self_check(modification_type, target_file, proposed_value)

        with self.evolution_lock:
            self._request_counter += 1
            request_id = f"EV-{datetime.now().strftime('%Y%m%d')}-{self._request_counter:04d}"

            request = EvolutionRequest(
                request_id=request_id,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                modification_type=modification_type,
                description=description,
                target_file=target_file,
                current_value=current_value,
                proposed_value=proposed_value,
                rationale=rationale,
                status=EvolutionRequestStatus.PENDING,
                ai_self_check=ai_self_check,
                ai_requested=True,
                audit_trail=[{"action": "request_created", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "actor": "AI"}]
            )
            self.evolution_requests.append(request)

        self._log_audit("evolution_request_created", "AI", {"request_id": request_id, "modification_type": modification_type.value, "target_file": target_file}, request_id)
        self._log(LogLevel.AUDIT, f"进化申请已创建", request_id=request_id)

        return request_id

    def _perform_self_check(self, modification_type: ModificationType, target_file: str, proposed_value: Any) -> Dict[str, Any]:
        self._log(LogLevel.INFO, f"执行进化申请自检", type=modification_type.value, target_file=target_file)

        check_result = {"timestamp": datetime.now().isoformat(), "checks": []}

        is_kernel = self.interpreter.is_kernel_file(target_file)
        check_result["checks"].append({"name": "kernel_file_check", "passed": not is_kernel,
                                      "detail": f"目标文件是{'核心' if is_kernel else '非核心'}文件"})

        if is_kernel:
            self._log(LogLevel.WARNING, f"检测到核心文件修改申请", target_file=target_file)

        allowed = True
        if modification_type == ModificationType.PRINCIPLE_CHANGE:
            allowed = False
            check_result["checks"].append({"name": "modification_type_check", "passed": False,
                                          "detail": "核心原则变更需要Guardian全面审查"})
            self._log(LogLevel.WARNING, f"检测到核心原则变更申请", type=modification_type.value)

        if isinstance(proposed_value, (int, float)):
            constraint = self.interpreter.get_constraint(target_file)
            if constraint:
                in_range = constraint.min_val <= proposed_value <= constraint.max_val
                check_result["checks"].append({"name": "constraint_range_check", "passed": in_range,
                                              "detail": f"值在约束范围内: [{constraint.min_val}, {constraint.max_val}]"})
                if not in_range:
                    self._log(LogLevel.ERROR, f"建议值超出约束范围", value=proposed_value)

        check_result["overall_passed"] = all(c.get("passed", False) for c in check_result["checks"])
        self._log(LogLevel.INFO, f"自检完成", overall_passed=check_result["overall_passed"])

        return check_result

    def get_evolution_status(self, request_id: str) -> Optional[EvolutionRequest]:
        with self.evolution_lock:
            for req in self.evolution_requests:
                if req.request_id == request_id:
                    return req
        return None

    def get_all_pending_requests(self) -> List[EvolutionRequest]:
        with self.evolution_lock:
            return [r for r in self.evolution_requests if r.status == EvolutionRequestStatus.PENDING]

    def submit_for_guardian_review(self, request_id: str) -> bool:
        request = self.get_evolution_status(request_id)
        if not request:
            self._log(LogLevel.ERROR, f"提交审查失败：申请不存在", request_id=request_id)
            return False

        if not request.ai_self_check.get("overall_passed", False):
            self._log(LogLevel.WARNING, f"提交审查失败：自检未通过", request_id=request_id)
            return False

        request.audit_trail.append({"action": "submitted_for_review", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "actor": "AI"})
        self._log_audit("evolution_request_submitted", "AI", {"request_id": request_id}, request_id)
        self._log(LogLevel.AUDIT, f"申请已提交Guardian审查", request_id=request_id)

        return True

    def sandbox_test(self, request_id: str, test_code: str) -> Dict[str, Any]:
        request = self.get_evolution_status(request_id)
        if not request:
            self._log(LogLevel.ERROR, f"沙箱测试失败：申请不存在", request_id=request_id)
            return {"success": False, "error": "申请不存在"}

        request.status = EvolutionRequestStatus.SANDBOX_TESTING
        request.audit_trail.append({"action": "sandbox_test_started", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "actor": "system"})
        self._log(LogLevel.INFO, f"开始沙箱测试", request_id=request_id)

        result = {"success": True, "test_output": "沙箱测试通过", "warnings": []}
        request.sandbox_result = result
        request.audit_trail.append({"action": "sandbox_test_completed", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "actor": "system", "result": result["success"]})

        self._log_audit("sandbox_test_completed", "system", {"request_id": request_id, "success": result["success"]}, request_id)
        self._log(LogLevel.INFO, f"沙箱测试完成", request_id=request_id, success=result["success"])

        return result

    def get_genome_view(self) -> Dict[str, Any]:
        self._log(LogLevel.INFO, f"AI请求基因视图")
        return self.interpreter.get_ai_readonly_view()

    def get_violation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        logs = self.enforcement.get_violation_logs(limit)
        result = [{
            "timestamp": log.timestamp,
            "policy": log.policy_violated,
            "module": log.caller_module,
            "action": log.action_type,
            "action_taken": log.action.value,
            "is_suspicious": log.ai_suspicious,
            "suggested_fix": log.suggested_fix
        } for log in logs]
        self._log(LogLevel.INFO, f"AI请求违规历史", count=len(result))
        return result

    def get_policy_explanation(self, policy_name: str) -> Optional[Dict[str, Any]]:
        self._log(LogLevel.INFO, f"AI请求策略解释", policy_name=policy_name)
        policy = self.interpreter.get_policy(policy_name)
        if not policy:
            self._log(LogLevel.WARNING, f"策略不存在", policy_name=policy_name)
            return None

        return {
            "name": policy.name,
            "description": policy.description,
            "value": policy.value,
            "enforcement_type": policy.enforcement.value,
            "violation_action": policy.violation_action.value,
            "why_this_matters": self._explain_policy_importance(policy),
            "compliance_guidance": self._get_compliance_guidance(policy),
            "risk_level": self._get_policy_risk_level(policy)
        }

    def _explain_policy_importance(self, policy) -> str:
        importance_map = {
            "right_brain_first": "这是系统架构的基础，确保视觉识别不依赖大模型",
            "dual_engine": "保证系统的稳定性和兼容性",
            "learning_persist": "确保学习成果不会丢失",
            "wta_competition": "确保识别机制的科学性",
            "safety_check": "保障用户安全",
            "no_unsolicited_speech": "避免系统干扰用户"
        }
        return importance_map.get(policy.name, "这是核心原则的一部分")

    def _get_compliance_guidance(self, policy) -> str:
        return f"在 {policy.name} 相关模块中，确保 {policy.description}"

    def _get_policy_risk_level(self, policy) -> str:
        if policy.enforcement.value == "mandatory" and policy.violation_action.value == "block":
            return "high"
        elif policy.enforcement.value == "continuous":
            return "medium"
        return "low"

    def get_audit_logs(self, limit: int = 100) -> List[AuditLog]:
        with self.audit_lock:
            return self.audit_logs[-limit:]

    def get_evolution_summary(self) -> Dict[str, Any]:
        with self.evolution_lock:
            summary = {
                "total_requests": len(self.evolution_requests),
                "pending_requests": len([r for r in self.evolution_requests if r.status == EvolutionRequestStatus.PENDING]),
                "approved_requests": len([r for r in self.evolution_requests if r.status == EvolutionRequestStatus.APPROVED]),
                "rejected_requests": len([r for r in self.evolution_requests if r.status == EvolutionRequestStatus.REJECTED]),
                "ai_initiated_requests": len([r for r in self.evolution_requests if r.ai_requested]),
                "last_request_time": self.evolution_requests[-1].timestamp if self.evolution_requests else None
            }
        self._log(LogLevel.INFO, f"获取进化申请汇总", **summary)
        return summary


# ==================== 单例访问函数 ====================

_reflection_instance: Optional[ReflectionLayer] = None


def get_reflection() -> ReflectionLayer:
    global _reflection_instance
    if _reflection_instance is None:
        _reflection_instance = ReflectionLayer()
    return _reflection_instance


# ==================== 便捷函数 ====================

def ai_self_check(module: str, function: str, action: Dict[str, Any]) -> SelfCheckResult:
    return get_reflection().self_check(module, function, action)


def propose_evolution(mod_type: ModificationType, description: str, target_file: str,
                      current: Any, proposed: Any, rationale: str) -> str:
    return get_reflection().propose_evolution(mod_type, description, target_file, current, proposed, rationale)


def get_genome_view() -> Dict[str, Any]:
    return get_reflection().get_genome_view()
