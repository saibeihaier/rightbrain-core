"""
对话管理器
管理整段会话的生命周期。

规则：
1. 系统不自言自语——只在用户说话时回答
2. 新物体出现时，后台调用左脑猜测，结果存着不说话
3. 用户问"这是什么"→如果有猜测结果→"我觉得是XX"
4. 用户教"这是香蕉"→存经验
5. 普通对话→调用左脑回复
"""
import json
import time

_conversation_history = []
MAX_HISTORY = 30
_visual_context = "当前没有识别到明显物体。"

# 新物体状态
_pending_new = None       # {marks, desc, time, learned}
_pending_guess = None     # {guessed_name, question, exp, time} — 左脑猜测结果
_pending_timeout = 300    # 5分钟超时
_last_new_notify_time = 0


def set_visual_context(context_str: str):
    global _visual_context
    _visual_context = context_str


def get_visual_context() -> str:
    return _visual_context


# ============ 新物体检测 ============

def mark_new_object(marks: dict):
    """标记一个新物体。不自言自语。"""
    global _pending_new, _last_new_notify_time
    now = time.time()
    color = marks.get('颜色', '')
    shape = marks.get('形状', '')
    desc = f"{color}{shape}" if color and shape else "未知物体"

    if _pending_new and _pending_new.get('desc') == desc and now - _pending_new.get('time', 0) < 60:
        return
    if _pending_new and _pending_new.get('learned'):
        return

    _pending_new = {"marks": marks, "desc": desc, "time": now, "learned": False}
    _pending_guess = None  # 新物体出现,旧的猜测作废
    
    if now - _last_new_notify_time > 30:
        print(f"[静默] 发现新物体：{desc}")
        _last_new_notify_time = now


def get_pending_new() -> dict | None:
    global _pending_new
    if _pending_new:
        if _pending_new.get('learned'):
            _pending_new = None; return None
        if time.time() - _pending_new.get('time', 0) > _pending_timeout:
            _pending_new = None; _pending_guess = None; return None
        return _pending_new
    return None


def mark_as_learned():
    """标记待学习物体已完成学习"""
    global _pending_new, _pending_guess
    _pending_new = None
    _pending_guess = None


# ============ 左脑猜测（不自言自语） ============

def set_guess_result(guess: dict):
    """
    设置左脑猜测结果。由后台线程调用。
    guess: {guessed_name, question, exp}
    存着但不说话，等用户问时再回答。
    """
    global _pending_guess
    if guess and guess.get('guessed_name'):
        _pending_guess = {
            "guessed_name": guess['guessed_name'],
            "question": guess.get('question', ''),
            "exp": guess.get('exp'),
            "time": time.time(),
        }
        print(f"[猜测] 左脑猜是{guess['guessed_name']}（等待用户询问）")


def get_guess_result() -> dict | None:
    """获取左脑猜测结果（5分钟内有效）"""
    global _pending_guess
    if _pending_guess:
        if time.time() - _pending_guess.get('time', 0) > 300:
            _pending_guess = None
            return None
        return _pending_guess
    return None


# ============ 对话处理 ============

def process_user_speech(user_text: str) -> str:
    """处理用户语音输入。返回回复文本。"""
    ctx = get_visual_context()
    text = user_text.strip()
    pending = get_pending_new()
    guess = get_guess_result()

    # === 判断意图 ===
    is_asking = any(kw in text for kw in ['这是什么', '这是啥', '什么东西', '那是什么', '看到什么'])
    is_teaching = text.startswith('这是') and len(text) > 3 and '什么' not in text

    # === 优先级1: 用户教"这是XXX" ===
    if is_teaching:
        name = text[2:].strip()
        if name:
            marks = pending.get('marks', {}) if pending else {}
            exp = {
                "name": name,
                "condition": {
                    "颜色": marks.get("颜色", "未知"),
                    "形状": marks.get("形状", "未知"),
                    "大小": marks.get("大小", "中"),
                } if marks else {},
                "action": f"这是{name}",
                "confidence": 0.8 if marks else 0.7,
            }
            mark_as_learned()
            return f"__LEARN__:{name}:{json.dumps(exp, ensure_ascii=False)}"

    # === 优先级2: 用户问"这是什么" ===
    if is_asking:
        if guess:
            # 有左脑猜测结果
            return f"我觉得是{guess['guessed_name']}，对吗？"
        elif pending:
            # 有待学习物体但没有猜测结果
            return f"我不知道这是{pending['desc']}，你教我这是什么？"
        else:
            return "我没有看到特别的东西。"

    # === 优先级3: 用户直接告知名称（"香蕉"）===
    if pending and not is_asking and not is_teaching:
        if 2 <= len(text) <= 10:
            marks = pending.get('marks', {})
            exp = {
                "name": text,
                "condition": {
                    "颜色": marks.get("颜色", "未知"),
                    "形状": marks.get("形状", "未知"),
                    "大小": marks.get("大小", "中"),
                },
                "action": f"这是{text}",
                "confidence": 0.8,
            }
            mark_as_learned()
            return f"__LEARN__:{text}:{json.dumps(exp, ensure_ascii=False)}"

    # === 优先级4: 用户确认左脑猜测 ===
    if guess:
        confirm = any(w in text for w in ['是', '对', '嗯', '没错', '就是'])
        correct = any(w in text for w in ['不是', '不对', '错了'])
        if confirm:
            exp = guess.get('exp')
            if exp:
                mark_as_learned()
                return f"__LEARN__:{guess['guessed_name']}:{json.dumps(exp, ensure_ascii=False)}"
        if correct and pending:
            return f"那这是什么？"

    # === 优先级5: 普通对话 ===
    return _normal_conversation(text, ctx)


def _normal_conversation(user_text: str, ctx: str) -> str:
    # 先检查技能库
    try:
        from rightbrain.learning.skill_manager import SkillManager
        if not hasattr(_normal_conversation, '_skill_mgr'):
            _normal_conversation._skill_mgr = SkillManager()
        skill_mgr = _normal_conversation._skill_mgr
        skill_reply = skill_mgr.match_and_execute(user_text)
        if skill_reply:
            return skill_reply
    except Exception:
        pass
    
    # 再检查是否可以学习新概念
    try:
        from rightbrain.learning.active_learner import ActiveLearner
        if hasattr(_normal_conversation, '_learner') and _normal_conversation._learner is not None:
            learner = _normal_conversation._learner
            # 如果用户提到了可能的物体名称（2-4个字的名词），尝试学习
            import re
            nouns = re.findall(r'[\u4e00-\u9fa5]{2,4}', user_text)
            for n in nouns:
                if n in ['什么', '这个', '那个', '自己', '大家', '一下', '一个']:
                    continue
                # 检查经验库中是否有（安全访问全局变量）
                try:
                    _mem_list = globals().get('memory_list', []) or []
                    _mem2_list = globals().get('memory2_list', []) or []
                    in_old = any(e.get('name') == n for e in _mem_list) if _mem_list else False
                    in_new = any(e.name == n for e in _mem2_list) if _mem2_list else False
                    if not in_old and not in_new:
                        learner.learn_concept(n)
                        break
                except Exception:
                    # 学习失败不影响主流程
                    pass
    except Exception:
        pass
    
    # 最后调用大模型
    try:
        from rightbrain.llm.bridge import ask_question
    except ImportError:
        ask_question = None
    
    try:
        history_str = ""
        recent = _conversation_history[-6:] if _conversation_history else []
        if recent:
            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in recent]) + "\n"

        prompt = f"""你是一个智能视觉助手。摄像头看到的场景：{ctx}

历史对话：
{history_str}
用户说：「{user_text}」

请根据视觉场景用中文给出简洁、自然的回复（不超过40字）。"""
        response = ask_question(prompt, temperature=0.7, max_tokens=150)
        cleaned = response.strip()
        for prefix in ["思考：", "思考:", "分析：", "分析:", "回答：", "回答:", "回复：", "回复:"]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        return cleaned if cleaned else "嗯，我在听呢。"
    except Exception as e:
        print(f"[对话] 调用大模型失败: {e}")
        return "嗯，我在听呢。"

# 模块级别变量，用于对话中的概念学习
memory_list = []
memory2_list = []

def set_memory_refs(mem, mem2):
    """设置经验库引用，用于主动学习"""
    global memory_list, memory2_list
    memory_list = mem.experiences if mem else []
    memory2_list = mem2.experiences if mem2 else []


def add_to_history(role: str, content: str):
    _conversation_history.append({"role": role, "content": content})
    if len(_conversation_history) > MAX_HISTORY:
        _conversation_history[:] = _conversation_history[-MAX_HISTORY:]


def get_history() -> list:
    return _conversation_history[-MAX_HISTORY:]


def clear_history():
    _conversation_history.clear()
