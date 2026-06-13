"""
genome.py — RightBrain 基因入口（v2）

这是项目的 DNA 入口点。只负责：
1. 基因完整性校验
2. 版本管理
3. 加载基因解释器

实际规则定义在：
- genome_spec.json (Layer 1: 声明层)
- interpreter.py (Layer 2: 解释器)
- enforcement.py (Layer 3: 强制执行层)
- reflection.py (Layer 4: 自检层)

核心原则（声明式）定义在 genome_spec.json，不可直接执行。
算法逻辑定义在 genome_algorithms.py 中，与规则分离。
"""

import hashlib
import json
import os
import sys

GENOME_VERSION = "2.0.0"  # v2: 四层架构

# ============================================================
# 旧版本兼容：保留CORE_PRINCIPLES和CORE_CONSTRAINTS
# 但现在它们从genome_spec.json加载，不再在代码中硬编码
# ============================================================

# 延迟导入，避免循环依赖
_genome_spec = None

def _get_genome_spec():
    """延迟加载基因组声明"""
    global _genome_spec
    if _genome_spec is None:
        spec_path = os.path.join(os.path.dirname(__file__), "genome_spec.json")
        with open(spec_path, 'r', encoding='utf-8') as f:
            _genome_spec = json.load(f)
    return _genome_spec


# 为兼容旧代码，提供直接访问方式
# 注意：现在从genome_spec.json动态加载，不再在代码中硬编码

def _load_spec():
    """延迟加载基因组声明"""
    global _genome_spec
    if _genome_spec is None:
        spec_path = os.path.join(os.path.dirname(__file__), "genome_spec.json")
        with open(spec_path, 'r', encoding='utf-8') as f:
            _genome_spec = json.load(f)
    return _genome_spec


# 提供兼容访问（通过函数获取）
def get_core_principles():
    """获取核心原则（从spec加载）"""
    return _load_spec().get("core_principles", {})


def get_core_constraints():
    """获取核心约束（从spec加载）"""
    return _load_spec().get("core_constraints", {})


# ============================================================
# 基因校验（核心职责）
# ============================================================

def get_genome_hash():
    """
    计算基因Spec的SHA256哈希

    注意：现在计算的是 genome_spec.json 的哈希，
    而不是 genome.py 的哈希。
    """
    spec_path = os.path.join(os.path.dirname(__file__), "genome_spec.json")
    with open(spec_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def check_genome_integrity():
    """
    启动时校验基因完整性。

    检查逻辑：
    1. 计算 genome_spec.json 的哈希
    2. 与 genome_checksum.json 比对
    3. 版本不匹配时自动更新校验文件
    """
    # 先尝试加载新版本的校验文件
    hash_file = os.path.join(os.path.dirname(__file__), "genome_checksum.json")

    if not os.path.exists(hash_file):
        # 可能是v1版本，先尝试v1格式
        v1_hash_file = os.path.join(os.path.dirname(__file__), "genome_checksum.json")
        if not os.path.exists(v1_hash_file):
            _write_checksum(hash_file, get_genome_hash())
            print(f"[基因] ✅ 校验文件已创建 (v{GENOME_VERSION})")
            return True

    try:
        with open(hash_file, 'r') as f:
            expected = json.load(f)
    except Exception:
        print("[基因] ❌ 校验文件损坏，重新创建")
        _write_checksum(hash_file, get_genome_hash())
        return True

    # 版本检查
    stored_version = expected.get("version", "1.0.0")
    if stored_version != GENOME_VERSION:
        print(f"[基因] ⬆️ 基因版本升级: {stored_version} → {GENOME_VERSION}")
        _write_checksum(hash_file, get_genome_hash())
        return True

    # 哈希检查
    current_hash = get_genome_hash()
    if current_hash != expected.get("hash"):
        print("[基因] ❌ 基因Spec已被修改但版本未升级！拒绝启动。")
        print(f"      期望版本: v{expected.get('version')}")
        print(f"      genome_spec.json 哈希已变更。")
        print(f"      请通过 ReflectionLayer.propose_evolution() 提出修改申请。")
        return False

    print(f"[基因] ✅ 校验通过 (v{GENOME_VERSION})")
    return True


def _write_checksum(path, hash_val):
    """写入校验文件"""
    import os as _os
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({
            "version": GENOME_VERSION,
            "hash": hash_val,
            "spec_hash": get_genome_hash()
        }, f, indent=2)


def get_version():
    """获取当前基因版本"""
    return GENOME_VERSION


# ============================================================
# 启动自检（v2: 验证所有Layer正常加载）
# ============================================================

def _check_layers():
    """检查所有Layer是否正常加载"""
    try:
        from . import interpreter
        from . import enforcement
        from . import reflection

        # 验证解释器
        interp = interpreter.get_interpreter()
        print(f"[基因] Layer 2 (Interpreter): ✅ v{interp.get_version()}")

        # 验证强制执行层
        enf = enforcement.get_enforcement()
        print(f"[基因] Layer 3 (Enforcement): ✅ 运行中")

        # 验证自检层
        refl = reflection.get_reflection()
        print(f"[基因] Layer 4 (Reflection): ✅ 就绪")

        return True
    except Exception as e:
        print(f"[基因] ❌ Layer加载失败: {e}")
        return False


# ============================================================
# 便捷访问函数
# ============================================================

def get_interpreter():
    """获取基因解释器"""
    from .interpreter import get_interpreter as _get
    return _get()


def get_enforcement():
    """获取强制执行层"""
    from .enforcement import get_enforcement as _get
    return _get()


def get_reflection():
    """获取自检层"""
    from .reflection import get_reflection as _get
    return _get()


def get_ai_view():
    """
    获取AI只读视图

    AI应该通过此方法获取基因规则，不能直接访问genome_spec.json
    """
    return get_reflection().get_genome_view()


# ============================================================
# 启动入口
# ============================================================

if not check_genome_integrity():
    sys.exit(1)

if not _check_layers():
    print("[基因] ⚠️ 部分Layer加载异常，系统继续运行但可能不稳定")
