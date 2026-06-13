import json
import time
from rightbrain.config import Config

DEBUG_MODE = Config.DEBUG_MODE

# 从 Config 读取特征权重和配置
FEATURE_WEIGHTS = getattr(Config, 'FEATURE_WEIGHTS', {
    '颜色': 0.35,
    '形状': 0.35,
    '大小': 0.12,
    '纹理': 0.08,
    '距离': 0.05,
    '边缘距离': 0.03,
    '支撑面': 0.02,
})

FORGET_THRESHOLD_DAYS = Config.FORGET_THRESHOLD_DAYS
FORGET_DECAY_RATE = Config.FORGET_DECAY_RATE
WEIGHT_ADJUSTMENT_RATE = Config.WEIGHT_ADJUSTMENT_RATE
MIN_WEIGHT = Config.MIN_WEIGHT
MAX_WEIGHT = Config.MAX_WEIGHT

SHAPE_SIMILARITY = Config.SHAPE_SIMILARITY


def _shape_similar(shape1, shape2):
    """判断两个形状是否相似"""
    if shape1 == shape2:
        return True
    if shape1 in SHAPE_SIMILARITY:
        return shape2 in SHAPE_SIMILARITY[shape1]
    if shape2 in SHAPE_SIMILARITY:
        return shape1 in SHAPE_SIMILARITY[shape2]
    return False


def _log(level, message):
    """日志输出函数"""
    if DEBUG_MODE:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] [Memory] {message}")

class ExperienceMemory:
    def __init__(self, json_path=None):
        self.experiences = []
        self.json_path = json_path
        self.weights = FEATURE_WEIGHTS.copy()  # 可调整的权重副本
        self.feedback_stats = {}  # 反馈统计：{'特征名': {'correct': 0, 'wrong': 0}}
        _log("INFO", "========== 经验记忆模块初始化 ==========")
        if json_path:
            _log("DEBUG", f"尝试加载经验文件: {json_path}")
            self.load(json_path)
        _log("INFO", f"当前经验库大小: {len(self.experiences)} 条")
        _log("DEBUG", f"特征权重配置: {self.weights}")
    
    def load(self, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8-sig') as f:
                self.experiences = json.load(f)
            # 确保每个经验都有last_matched_time字段
            current_time = time.time()
            for exp in self.experiences:
                if 'last_matched_time' not in exp:
                    exp['last_matched_time'] = current_time
            _log("INFO", f"成功加载 {len(self.experiences)} 条经验")
        except Exception as e:
            _log("ERROR", f"加载经验文件失败: {str(e)}")
            self.experiences = []
    
    def save(self, json_path):
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.experiences, f, indent=2, ensure_ascii=False)
            _log("INFO", f"成功保存 {len(self.experiences)} 条经验到 {json_path}")
        except Exception as e:
            _log("ERROR", f"保存经验文件失败: {str(e)}")
    
    def match(self, marks_dict, threshold=0.6):
        _log("INFO", "========== 经验匹配开始 ==========")
        _log("DEBUG", f"输入标记: {marks_dict}")
        _log("DEBUG", f"匹配阈值: {threshold}")
        
        # 优先使用深度传感器的检测结果（使用统一的字段名）
        deep_class_raw = marks_dict.get('_yolo_class', marks_dict.get('_deep_class', ''))
        deep_class = deep_class_raw.lower() if deep_class_raw else ''
        deep_confidence = marks_dict.get('_yolo_confidence', marks_dict.get('_deep_confidence', 0))
        
        if deep_class and deep_confidence > 0.3:
            _log("INFO", f"检测到深度传感器结果: {deep_class} (置信度: {deep_confidence:.2f})")
            
            # 使用统一的 YOLO 类别映射
            chinese_name = Config.YOLO_CLASS_MAPPING.get(deep_class, deep_class)
            
            # 先尝试匹配经验库中名称相同的经验
            matched_exps = [exp for exp in self.experiences 
                          if exp.get('name') == chinese_name]
            
            if matched_exps:
                best_exp = matched_exps[0]
                # 使用深度传感器的置信度作为匹配度
                score = min(deep_confidence + 0.2, 0.95)  # 适当提高置信度
                _log("INFO", f"✅ 深度传感器匹配成功: {chinese_name} (匹配度: {score:.2f})")
                best_exp['last_matched_time'] = time.time()
                return (score, best_exp.get('action', f'这是{chinese_name}'), best_exp)
            else:
                _log("INFO", f"深度传感器检测到 {chinese_name}，但经验库中没有匹配的经验")
        
        # 优先使用深度类别匹配手机
        yolo_class = marks_dict.get('_yolo_class', '')
        yolo_conf_val = marks_dict.get('_yolo_confidence', 0)
        yolo_conf_num = float(yolo_conf_val) if isinstance(yolo_conf_val, str) else yolo_conf_val
        if yolo_class == 'cell phone' and yolo_conf_num > 0.5:
            phone_exps = [exp for exp in self.experiences if exp.get('name') == '手机']
            if phone_exps:
                phone_exp = phone_exps[0]
                _log("INFO", f"深度检测到手机，优先匹配")
                phone_exp['last_matched_time'] = time.time()
                return (phone_exp.get('confidence', 0.7), phone_exp.get('action', '这是手机'), phone_exp)
        
        # 如果没有深度类别，再回退到宽泛的长方形规则（但增加额外约束）
        is_handheld_phone_candidate = (
            marks_dict.get('形状') == '长方形' and
            (marks_dict.get('is_handheld', False) or marks_dict.get('深度类别') in ['cell phone', 'phone', 'mobile phone'])
        )
        
        if is_handheld_phone_candidate:
            # 增加排除条件：颜色不能是肤色（避免人脸误判），且大小为中等
            color = marks_dict.get('颜色', '')
            obj_size = marks_dict.get('大小', '中')
            if color != '肤色' and obj_size == '中':
                _log("INFO", "检测到长方形中等大小非肤色物体，可能是手机")
                phone_exps = [exp for exp in self.experiences 
                             if exp.get('name') == '手机' and exp.get('priority', 0) > 0]
                if phone_exps:
                    phone_exp = phone_exps[0]
                    _log("INFO", f"找到手机经验，优先返回: {phone_exp.get('name')}")
                    phone_exp['last_matched_time'] = time.time()
                    return (phone_exp.get('confidence', 0.5), phone_exp.get('action', '未知'), phone_exp)
        
        # 特殊处理：对于人脸检测，如果提供了年龄和性别
        # 优先匹配包含这些特征的更具体的经验
        is_face_with_age_gender = (
            marks_dict.get('形状') == '人脸' and
            '年龄' in marks_dict and
            '性别' in marks_dict and
            marks_dict.get('年龄') != '未知' and
            marks_dict.get('性别') != '未知'
        )
        
        # 如果是人脸且有年龄性别，先尝试匹配具体的经验
        if is_face_with_age_gender:
            _log("INFO", "检测到人脸且有年龄性别信息，优先匹配具体经验")
            specific_exps = []
            generic_exps = []
            
            for exp in self.experiences:
                cond = exp.get('condition', {})
                if cond.get('形状') == '人脸':
                    if '年龄' in cond and '性别' in cond:
                        specific_exps.append(exp)
                    else:
                        generic_exps.append(exp)
            
            # 先在具体经验中找最佳匹配
            best_exp, best_score = self._find_best_match(marks_dict, specific_exps)
            
            # 如果具体经验匹配度不够高，再考虑通用经验
            if best_exp and best_score >= 0.85:
                _log("INFO", f"找到具体匹配: {best_exp.get('name')}, 匹配度: {best_score:.2f}")
            else:
                # 在通用经验中找最佳匹配
                generic_exp, generic_score = self._find_best_match(marks_dict, generic_exps)
                
                # 如果通用经验匹配度更高或具体经验匹配度不够，使用通用经验
                if generic_exp and (not best_exp or generic_score > best_score or best_score < 0.70):
                    best_exp = generic_exp
                    best_score = generic_score
                    _log("INFO", f"回退到通用经验: {best_exp.get('name')}, 匹配度: {best_score:.2f}")
        else:
            # 普通匹配
            best_exp, best_score = self._find_best_match(marks_dict, self.experiences)
        
        # 更新最佳匹配结果
        if best_score >= threshold and best_exp:
            best_exp['last_matched_time'] = time.time()
            action = best_exp.get('action', '未知')
            _log("INFO", f"匹配成功! 最佳匹配度: {best_score:.4f}, 行动: {action}")
        else:
            best_exp = None
            action = "未找到匹配经验"
        
        return (best_score, action, best_exp) if best_exp else (best_score, action, None)
    
    def _find_best_match(self, marks_dict, experiences):
        """在给定经验列表中找最佳匹配"""
        best_exp = None
        best_score = 0.0
        
        for idx, exp in enumerate(experiences):
            cond = exp["condition"]
            matched_weight = 0.0
            total_weight = 0.0
            
            # 判断是否为人脸经验
            is_face_exp = cond.get('形状') == '人脸'
            is_face_input = marks_dict.get('形状') == '人脸'
            
            for key, val in cond.items():
                # 对于人脸匹配，降低颜色权重，提高年龄性别权重
                if is_face_exp and is_face_input:
                    if key == '颜色':
                        weight = 0.05  # 人脸匹配时颜色权重降低
                    elif key in ['年龄', '性别']:
                        weight = 0.3   # 人脸匹配时年龄性别权重提高
                    elif key == '形状':
                        weight = 0.3   # 形状权重
                    else:
                        weight = self.weights.get(key, 0.1)
                else:
                    weight = self.weights.get(key, 0.1)
                
                total_weight += weight
                
                if key in marks_dict:
                    mark_val = marks_dict[key]
                    if mark_val == val:
                        matched_weight += weight
                        _log("DEBUG", f"    特征 '{key}' 匹配 (权重:{weight:.2f})")
                    elif key == '形状' and _shape_similar(mark_val, val):
                        matched_weight += weight * 0.7
                        _log("DEBUG", f"    特征 '{key}' 相似匹配 (权重:{weight:.2f} × 0.7)")
                    else:
                        _log("DEBUG", f"    特征 '{key}' 不匹配 (权重:{weight:.2f})")
                else:
                    _log("DEBUG", f"    特征 '{key}' 不匹配 (权重:{weight:.2f})")
            
            score = matched_weight / total_weight if total_weight > 0 else 0
            _log("DEBUG", f"经验#{idx+1}: 条件={cond}, 匹配度={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_exp = exp
        
        return best_exp, best_score
    
    def find_experience(self, marks_dict):
        """根据标记找到匹配的经验（用于用户反馈）"""
        best_score, action, best_exp = self.match(marks_dict, threshold=0.0)
        if best_score > 0.5:
            return best_exp
        return None
    
    def add_or_update(self, new_exp):
        _log("INFO", "========== 添加/更新经验 ==========")
        _log("DEBUG", f"新经验: {new_exp}")
        
        new_cond = new_exp.get("condition", {})
        
        # 判断是否为人脸经验
        is_face = new_cond.get("形状") == "人脸"
        
        for idx, exp in enumerate(self.experiences):
            old_cond = exp.get("condition", {})
            
            # 精确匹配条件
            if old_cond == new_cond:
                old_action = exp["action"]
                old_name = exp.get("name", "未知")
                old_confidence = exp.get("confidence", 0.5)
                new_confidence = max(old_confidence, new_exp.get("confidence", 0.5))
                
                exp["action"] = new_exp["action"]
                exp["confidence"] = new_confidence
                if "name" in new_exp:
                    exp["name"] = new_exp["name"]
                
                _log("INFO", f"更新现有经验 #{idx+1}（精确匹配）")
                _log("DEBUG", f"  原名称: {old_name} -> 新名称: {exp.get('name', '未知')}")
                _log("DEBUG", f"  原行动: {old_action} -> 新行动: {exp['action']}")
                _log("DEBUG", f"  原置信度: {old_confidence} -> 新置信度: {new_confidence}")
                
                if self.json_path:
                    self.save(self.json_path)
                return False
            
            # 对于人脸，如果年龄和性别相同，也认为是同一个人
            if is_face and old_cond.get("形状") == "人脸":
                same_age = old_cond.get("年龄") == new_cond.get("年龄")
                same_gender = old_cond.get("性别") == new_cond.get("性别")
                
                if same_age and same_gender:
                    old_action = exp["action"]
                    old_name = exp.get("name", "未知")
                    old_confidence = exp.get("confidence", 0.5)
                    new_confidence = max(old_confidence, new_exp.get("confidence", 0.5))
                    
                    exp["action"] = new_exp["action"]
                    exp["confidence"] = new_confidence
                    if "name" in new_exp:
                        exp["name"] = new_exp["name"]
                    # 更新条件（可能包含更多字段）
                    exp["condition"] = new_cond
                    
                    _log("INFO", f"更新现有经验 #{idx+1}（人脸匹配：年龄={new_cond.get('年龄')}, 性别={new_cond.get('性别')}）")
                    _log("DEBUG", f"  原名称: {old_name} -> 新名称: {exp.get('name', '未知')}")
                    _log("DEBUG", f"  原行动: {old_action} -> 新行动: {exp['action']}")
                    
                    if self.json_path:
                        self.save(self.json_path)
                    return False
        
        self.experiences.append(new_exp)
        _log("INFO", f"添加新经验, 当前经验库大小: {len(self.experiences)}")
        _log("DEBUG", f"  condition: {new_exp['condition']}")
        _log("DEBUG", f"  name: {new_exp.get('name', '未知')}")
        _log("DEBUG", f"  action: {new_exp['action']}")
        _log("DEBUG", f"  confidence: {new_exp.get('confidence', 0.5)}")
        
        if self.json_path:
            self.save(self.json_path)
        return True
    
    def update_confidence(self, condition, success, delta=0.1):
        _log("INFO", "========== 更新置信度 ==========")
        _log("DEBUG", f"条件: {condition}")
        _log("DEBUG", f"操作: {'成功' if success else '失败'}, 增量: {delta}")
        
        for idx, exp in enumerate(self.experiences):
            if exp["condition"] == condition:
                current_confidence = exp.get("confidence", 0.5)
                if success:
                    new_confidence = min(1.0, current_confidence + delta)
                else:
                    new_confidence = max(0.0, current_confidence - delta)
                
                exp["confidence"] = new_confidence
                
                _log("INFO", f"找到匹配经验 #{idx+1}, 置信度更新成功")
                _log("DEBUG", f"  更新前: {current_confidence:.4f} -> 更新后: {new_confidence:.4f}")
                
                if self.json_path:
                    self.save(self.json_path)
                return True
        
        _log("WARNING", "未找到匹配的经验, 置信度更新失败")
        return False
    
    def get_confidence(self, condition):
        for exp in self.experiences:
            if exp["condition"] == condition:
                confidence = exp.get("confidence", 0.5)
                _log("DEBUG", f"查询置信度 - 条件:{condition}, 置信度:{confidence}")
                return confidence
        _log("DEBUG", f"查询置信度 - 条件:{condition}, 未找到, 返回0.0")
        return 0.0
    
    def decay_unused_confidences(self, threshold_days=FORGET_THRESHOLD_DAYS, decay_rate=FORGET_DECAY_RATE):
        """
        遗忘机制：降低长时间未匹配经验的置信度
        
        Args:
            threshold_days: 超过多少天未匹配开始降低置信度
            decay_rate: 每次降低的置信度比例
        
        Returns:
            被降低置信度的经验数量
        """
        _log("INFO", "========== 遗忘机制开始 ==========")
        current_time = time.time()
        decayed_count = 0
        
        for exp in self.experiences:
            last_matched = exp.get('last_matched_time', current_time)
            days_since_matched = (current_time - last_matched) / (24 * 3600)
            
            if days_since_matched > threshold_days:
                old_confidence = exp.get('confidence', 0.5)
                new_confidence = max(0.1, old_confidence - decay_rate)
                exp['confidence'] = new_confidence
                decayed_count += 1
                _log("DEBUG", f"遗忘: {exp.get('name', '未知')} ({exp['condition']}), "
                            f"置信度: {old_confidence:.2f} -> {new_confidence:.2f}, "
                            f"已{days_since_matched:.1f}天未匹配")
        
        if decayed_count > 0 and self.json_path:
            self.save(self.json_path)
            _log("INFO", f"遗忘完成, {decayed_count} 条经验被降低置信度")
        else:
            _log("INFO", "没有需要遗忘的经验")
        
        _log("INFO", "========== 遗忘机制结束 ==========")
        return decayed_count
    
    def remove_exp(self, condition):
        """
        删除指定经验
        
        Args:
            condition: 要删除的经验条件
        
        Returns:
            是否成功删除
        """
        for idx, exp in enumerate(self.experiences):
            if exp["condition"] == condition:
                exp_name = exp.get('name', '未知')
                del self.experiences[idx]
                if self.json_path:
                    self.save(self.json_path)
                _log("INFO", f"已删除经验: [{exp_name}] {condition}")
                return True
        _log("WARNING", f"未找到要删除的经验: {condition}")
        return False
    
    def record_feedback(self, exp, is_correct):
        """
        记录用户反馈，用于动态调整权重
        
        Args:
            exp: 被评估的经验
            is_correct: 反馈是否正确
        """
        condition = exp.get('condition', {})
        for key, value in condition.items():
            if value:  # 只统计有值的特征
                if key not in self.feedback_stats:
                    self.feedback_stats[key] = {'correct': 0, 'wrong': 0}
                if is_correct:
                    self.feedback_stats[key]['correct'] += 1
                else:
                    self.feedback_stats[key]['wrong'] += 1
        self._adjust_weights()
    
    def _adjust_weights(self):
        """根据反馈统计动态调整特征权重"""
        if not self.feedback_stats:
            return
        
        total_correct = sum(stats['correct'] for stats in self.feedback_stats.values())
        if total_correct < 3:  # 至少需要3次正确反馈才调整
            return
        
        adjustments = {}
        for feature, stats in self.feedback_stats.items():
            if stats['correct'] + stats['wrong'] >= 2:
                accuracy = stats['correct'] / (stats['correct'] + stats['wrong'])
                current_weight = self.weights.get(feature, FEATURE_WEIGHTS.get(feature, 0.1))
                
                # 如果准确率高，增加权重；准确率低，降低权重
                if accuracy > 0.7:
                    new_weight = min(MAX_WEIGHT, current_weight + WEIGHT_ADJUSTMENT_RATE)
                    adjustments[feature] = (current_weight, new_weight, '增加')
                elif accuracy < 0.4:
                    new_weight = max(MIN_WEIGHT, current_weight - WEIGHT_ADJUSTMENT_RATE)
                    adjustments[feature] = (current_weight, new_weight, '降低')
        
        if adjustments:
            for feature, (old, new, direction) in adjustments.items():
                self.weights[feature] = new
                _log("DEBUG", f"权重调整: {feature}: {old:.3f} -> {new:.3f} ({direction})")
            
            _log("INFO", f"权重已更新: {self.weights}")
    
    def get_weights(self):
        """获取当前权重配置"""
        return self.weights.copy()
    
    def reset_weights(self):
        """重置权重为默认值"""
        self.weights = FEATURE_WEIGHTS.copy()
        self.feedback_stats = {}
        _log("INFO", "权重已重置为默认值")