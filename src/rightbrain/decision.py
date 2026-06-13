"""
决策引擎模块

职责：
1. identify() — 纯右脑匹配，返回 (action, score, matched_exp, is_new)
2. guess_new_object() — 只做右脑联想推理，不调大模型
3. guess_with_llm() — 只有明确需要时才调大模型猜
4. learn_experience() — 调用左脑学习（后备方案）
"""
import hashlib
import json
import re
import time
import threading
from rightbrain.config import Config
from rightbrain.cv_core.affect_sensor import apply_affect_to_action
try:
    from rightbrain.llm.bridge import call_left_brain, ask_question
except ImportError:
    call_left_brain = None
    ask_question = None

try:
    from rightbrain.utils.natural_language_generator import generate_response
    NLG_AVAILABLE = True
except ImportError:
    NLG_AVAILABLE = False

try:
    from rightbrain.learning.associative_memory import get_associative_memory
    ASSOCIATIVE_MEMORY_AVAILABLE = True
except ImportError:
    ASSOCIATIVE_MEMORY_AVAILABLE = False

LEARNING_CACHE = {}
LEARNING_CACHE_TTL = 60


def _generate_feature_hash(marks_dict):
    key_fields = ['颜色', '形状', '大小', '纹理', '光照']
    key_values = [str(marks_dict.get(f, 'unknown')) for f in key_fields]
    return hashlib.md5('|'.join(key_values).encode('utf-8')).hexdigest()


def _is_learning_in_progress(marks_dict):
    feature_hash = _generate_feature_hash(marks_dict)
    if feature_hash in LEARNING_CACHE:
        entry = LEARNING_CACHE[feature_hash]
        if time.time() - entry['timestamp'] < LEARNING_CACHE_TTL:
            return True, entry.get('action', '后台学习中...')
        else:
            del LEARNING_CACHE[feature_hash]
    return False, None


def _mark_learning_start(marks_dict):
    feature_hash = _generate_feature_hash(marks_dict)
    LEARNING_CACHE[feature_hash] = {'timestamp': time.time(), 'action': '正在学习...'}


def _mark_learning_complete(marks_dict, action):
    feature_hash = _generate_feature_hash(marks_dict)
    LEARNING_CACHE.pop(feature_hash, None)


def _clear_learning_cache():
    now = time.time()
    expired = [h for h, e in LEARNING_CACHE.items() if now - e['timestamp'] >= LEARNING_CACHE_TTL]
    for h in expired:
        del LEARNING_CACHE[h]


def identify(marks_dict, memory, confidence_threshold=0.5):
    """
    纯右脑识别，返回 (action, score, matched_exp, is_new)
    """
    from rightbrain.config import Config
    _clear_learning_cache()
    
    yolo_class = marks_dict.get('_yolo_class', '')
    yolo_conf = marks_dict.get('_yolo_confidence', 0.0)
    feeling = marks_dict.get('感受', '中性')
    
    score, action, best_exp = memory.match(marks_dict, threshold=confidence_threshold)
    
    yolo_min_conf = Config.clamp_to_genome("yolo_confidence", 0.6)
    min_conf = Config.clamp_to_genome("min_match_score", 0.4)
    if yolo_conf >= yolo_min_conf and (best_exp is None or score < min_conf):
        friendly_name = Config.YOLO_CLASS_MAPPING.get(yolo_class, yolo_class)
        action = friendly_name
        action = apply_affect_to_action(action, feeling)
        return action, yolo_conf, None, False
    
    if best_exp is not None:
        if NLG_AVAILABLE:
            try:
                action = generate_response(marks_dict, best_exp, is_new=False)
            except Exception:
                name = best_exp.get("name", "物体")
                exp_action = best_exp["action"]
                action = f"{name}：{exp_action}"
        else:
            name = best_exp.get("name", "物体")
            exp_action = best_exp["action"]
            action = f"{name}：{exp_action}"
        
        action = apply_affect_to_action(action, feeling)
        return action, score, best_exp, False
    
    color = marks_dict.get('颜色', '')
    shape = marks_dict.get('形状', '')
    desc = f"{color}{shape}" if color and shape else "未知物体"
    action = apply_affect_to_action(desc, feeling)
    return action, score, None, False


def guess_new_object(marks_dict, associative_memory=None):
    """
    右脑联想推理——不调大模型。
    
    只做：
    - 联-推理（在已有经验里找相似的）
    - 返回联想结果
    
    不调大模型。如果需要大模型猜测，外部调用 guess_with_llm()。
    
    返回 dict:
    {
        'guessed_name': str,      # 联想推理猜的名称（如果有）
        'action': str,
        'confidence': float,
        'question': str,
        'associations': list,     # 联想推理结果
        'exp': dict,              # 如果用户确认可直接存的经验
        'has_guess': bool,        # 是否有联想推理结果
    }
    """
    result = {
        'guessed_name': '',
        'action': '',
        'confidence': 0.0,
        'question': '',
        'associations': [],
        'exp': None,
        'has_guess': False,
    }
    
    color = marks_dict.get('颜色', '')
    shape = marks_dict.get('形状', '')
    desc = f"{color}{shape}" if color and shape else "未知物体"
    
    # 联想推理（纯右脑，不调大模型）
    if associative_memory is not None:
        try:
            analogical = associative_memory.analogical_reasoning(marks_dict)
            if analogical and analogical.get('success'):
                suggested = analogical.get('analogical_object', '')
                sim = analogical.get('similarity', 0.0)
                if suggested and sim > 0.3:
                    result['associations'] = [{
                        'type': '类比推理',
                        'source': f"相似于{suggested}",
                        'similarity': sim,
                        'action': analogical.get('suggested_action', ''),
                    }]
                    result['guessed_name'] = suggested
                    result['action'] = analogical.get('suggested_action', '')
                    result['confidence'] = sim
                    result['question'] = f"我看到一个{desc}，是{suggested}吗？"
                    result['has_guess'] = True
                    
                    # 构建经验对象
                    result['exp'] = {
                        "name": suggested,
                        "condition": {
                            "颜色": color,
                            "形状": shape,
                            "大小": marks_dict.get('大小', '中'),
                        },
                        "action": f"这是{suggested}",
                        "confidence": max(0.5, sim),
                        "priority": 10,
                    }
        except Exception as e:
            print(f"[决策] 联想推理失败: {e}")
    
    return result


def guess_with_llm(marks_dict, ask_image=False):
    """
    左脑猜物体——调大模型。只在明确需要时调用。
    
    Args:
        marks_dict: 右脑提取的特征
        ask_image: 是否传图片给大模型（仅在完全未知时）
    
    返回和 guess_new_object 相同结构。
    """
    result = {
        'guessed_name': '',
        'action': '',
        'confidence': 0.0,
        'question': '',
        'exp': None,
        'has_guess': False,
    }
    
    color = marks_dict.get('颜色', '')
    shape = marks_dict.get('形状', '')
    size = marks_dict.get('大小', '中')
    desc = f"{color}{shape}" if color and shape else "未知物体"
    
    try:
        feature_lines = [
            f"- 颜色: {color}", f"- 形状: {shape}", f"- 大小: {size}",
            f"- 纹理: {marks_dict.get('纹理', '未知')}",
            f"- 光照: {marks_dict.get('光照', '未知')}",
        ]
        
        roi = marks_dict.get('_roi_features', {})
        if roi:
            circ = roi.get('circularity', 0)
            if circ > 0.6:
                feature_lines.append(f"- 形状特点：圆形度{circ:.2f}(规则圆润)")
            elif circ > 0.3:
                feature_lines.append(f"- 形状特点：圆形度{circ:.2f}(有一定棱角)")
            else:
                ar = roi.get('aspect_ratio', 1)
                feature_lines.append(f"- 形状特点：圆形度{circ:.2f}，宽高比{ar}(细长/扁平)")
            tc = roi.get('texture_contrast', 0)
            if tc > 100: feature_lines.append(f"- 表面纹理：粗糙（对比度{tc:.0f}）")
            elif tc > 30: feature_lines.append(f"- 表面纹理：有一定细节（对比度{tc:.0f}）")
            else: feature_lines.append(f"- 表面纹理：光滑")
            em = roi.get('edge_mean', 0)
            if em > 20: feature_lines.append(f"- 边缘：清晰锐利（边缘强度{em:.0f}）")
            else: feature_lines.append(f"- 边缘：柔和")
        
        if marks_dict.get('年龄') and marks_dict.get('性别'):
            feature_lines.append(f"- 这是人脸：{marks_dict.get('年龄')} {marks_dict.get('性别')}")
        
        features_str = "\n".join(feature_lines)
        
        prompt = f"""你是一个智能体的视觉识别模块。摄像头看到一个新物体，视觉特征如下：
{features_str}

请根据特征猜测这是什么物体。只输出JSON，不要其他文字：
{{"name":"物体名称（中文，1-4字）","action":"简短行动描述","confidence":0.0-1.0}}"""
        
        roi_b64 = marks_dict.get('_roi_features', {}).get('roi_image_b64') if ask_image else None
        response_text = ask_question(prompt, temperature=0.3, max_tokens=200, image_b64=roi_b64)
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            guessed_name = parsed.get('name', '')
            guessed_action = parsed.get('action', '')
            confidence = float(parsed.get('confidence', 0.4))
        
            if guessed_name and len(guessed_name) <= 10:
                result['guessed_name'] = guessed_name
                result['action'] = guessed_action
                result['confidence'] = confidence
                result['question'] = f"我看到一个{desc}，是{guessed_name}吗？"
                result['has_guess'] = True
                result['exp'] = {
                    "name": guessed_name,
                    "condition": {"颜色": color, "形状": shape, "大小": size,},
                    "action": f"这是{guessed_name}",
                    "confidence": max(0.5, confidence),
                    "priority": 10,
                }
    except Exception as e:
        print(f"[决策] 左脑猜测失败: {e}")
    
    return result


def learn_experience(marks_dict, memory, blocking=False):
    """调用左脑学习并存入经验库（不常用）"""
    if _is_learning_in_progress(marks_dict)[0]:
        return None, False
    _mark_learning_start(marks_dict)
    try:
        new_exp = call_left_brain(marks_dict, 0.0, blocking=True)
        if not new_exp or not new_exp.get("action"):
            _mark_learning_complete(marks_dict, "学习失败")
            return None, False
        new_exp["pleasantness"] = marks_dict.get('愉悦度', 0.5)
        new_exp["feeling"] = marks_dict.get('感受', '中性')
        memory.add_or_update(new_exp)
        _mark_learning_complete(marks_dict, new_exp.get('action', ''))
        return new_exp, True
    except Exception as e:
        print(f"[决策] 左脑学习失败: {e}")
        _mark_learning_complete(marks_dict, "学习失败")
        return None, False
