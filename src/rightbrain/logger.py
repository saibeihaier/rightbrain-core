"""
审计日志模块 — 记录关键决策链，方便问题回溯

不侵入核心逻辑，只记录关键事件。
每条日志包含：时间戳、触发源、事件类型、详情。

使用方式：
    from rightbrain.logger import audit_log
    audit_log("curiosity", "左脑猜测", {"物体": "红圆形", "置信度": 0.4})
"""
import os
import json
import time
from datetime import datetime

_AUDIT_FILE = None
_MAX_LINES = 10000

# 日志文件路径（项目根目录下）
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(_LOG_DIR, "audit.log")


def audit_log(source: str, event: str, details: dict = None):
    """
    写入一条审计日志。
    
    Args:
        source: 触发源（如"curiosity", "dialogue", "learning", "safety", "speak"）
        event: 事件类型（如"左脑猜测", "用户教学", "安全检查"）
        details: 详情字典（会被转成 JSON 字符串）
    """
    try:
        record = {
            "t": datetime.now().strftime("%H:%M:%S"),
            "ts": time.time(),
            "src": source[:20],
            "evt": event[:40],
        }
        if details:
            record["d"] = _sanitize(details)
        
        line = json.dumps(record, ensure_ascii=False) + "\n"
        
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
        
        # 文件太大时截断
        _rotate_if_needed()
    except Exception:
        pass  # 审计日志本身不出错


def _sanitize(d):
    """确保可 JSON 序列化"""
    if isinstance(d, dict):
        return {str(k)[:30]: _sanitize(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_sanitize(x) for x in d[:10]]
    if isinstance(d, float):
        return round(d, 4)
    if hasattr(d, 'name'):
        return str(d.name)
    if hasattr(d, '__dict__'):
        return str(type(d).__name__)
    return str(d)[:200]


def _rotate_if_needed():
    """日志超过上限时截断前半部分"""
    try:
        if os.path.getsize(_LOG_FILE) > 1024 * 512:  # 512KB
            with open(_LOG_FILE, "r") as f:
                lines = f.readlines()
            if len(lines) > _MAX_LINES:
                with open(_LOG_FILE, "w") as f:
                    f.writelines(lines[-_MAX_LINES // 2:])
    except Exception:
        pass


def get_recent_logs(n: int = 50) -> list:
    """获取最近 n 条日志"""
    try:
        with open(_LOG_FILE, "r") as f:
            lines = f.readlines()
        return [json.loads(l) for l in lines[-n:]]
    except Exception:
        return []


def print_recent(n: int = 20):
    """打印最近 n 条日志"""
    for r in get_recent_logs(n):
        t = r.get("t", "")
        s = r.get("src", "")
        e = r.get("evt", "")
        d = r.get("d", {})
        detail = json.dumps(d, ensure_ascii=False) if d else ""
        print(f"[{t}] [{s}] {e} {detail}")
