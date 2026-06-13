#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联想推理模块 - 实现形象思维的联想推理能力
支持：
1. 特征关联推理：基于颜色、形状等特征的联想
2. 经验关联网络：建立经验之间的关联关系
3. 置信度强化：通过反馈逐步加强正确经验
4. 类比推理：基于相似经验进行推理
5. 链式联想：多层次联想推理，形成思维链条
6. 形象思维网络：将概念串联成有意义的思维网络
"""

import json
import time
import os
from typing import Dict, List, Tuple, Optional, Any, Set

DEBUG_MODE = True

# 特征关联映射 - 定义特征之间的联想关系（增强版）
FEATURE_ASSOCIATIONS = {
    # 颜色联想
    '颜色': {
        '红': ['热情', '温暖', '苹果', '西红柿', '草莓', '火焰', '危险', '爱情', '血液', '生命力', '警示', '喜庆'],
        '黄': ['明亮', '温暖', '香蕉', '柠檬', '向日葵', '阳光', '快乐', '希望', '活力', '秋天', '丰收'],
        '蓝': ['冷静', '天空', '海洋', '水', '科技', '信任', '智慧', '宁静', '深度', '自由', '梦想'],
        '绿': ['自然', '生命', '植物', '草地', '环保', '健康', '春天', '成长', '希望', '和平', '新鲜'],
        '橙': ['活力', '温暖', '橙子', '胡萝卜', '秋天', '创意', '热情', '社交', '能量'],
        '紫': ['神秘', '高贵', '葡萄', '茄子', '浪漫', '智慧', '魔法', '梦想', '艺术'],
        '黑': ['神秘', '夜晚', '正式', '优雅', '力量', '权威', '深度', '未知', '宇宙'],
        '白': ['纯洁', '干净', '雪', '云', '简约', '和平', '空白', '开始', '光明'],
        '灰': ['中性', '稳重', '金属', '石头', '成熟', '平衡', '低调'],
        '肤色': ['人类', '温暖', '生命', '亲近', '情感', '交流'],
    },
    # 形状联想
    '形状': {
        '圆形': ['完整', '和谐', '球', '太阳', '月亮', '眼睛', '水果', '团圆', '完美', '循环', '宇宙'],
        '椭圆形': ['蛋', '脸', '叶子', '橄榄球', '柔和', '自然', '流动'],
        '长方形': ['书本', '盒子', '建筑', '门', '屏幕', '手机', '平板', '电子设备', '稳定', '规则', '秩序'],
        '正方形': ['稳定', '盒子', '窗户', '瓷砖', '平衡', '对称', '坚固'],
        '三角形': ['稳定', '山峰', '箭头', '警告', '方向', '力量', '进取'],
        '人脸': ['人类', '情感', '交流', '身份', '记忆', '故事', '灵魂'],
        '不规则': ['自然', '独特', '艺术', '自由', '创意', '个性'],
    },
    # 大小联想
    '大小': {
        '大': ['重要', '显眼', '力量', '主导', '权威', '壮观', '震撼'],
        '中': ['普通', '适中', '常见', '平衡', '舒适', '实用'],
        '小': ['可爱', '精致', '细节', '次要', '珍贵', '亲密', '便携'],
    },
    # 纹理联想
    '纹理': {
        '光滑': ['舒适', '现代', '干净', '玻璃', '金属', '科技', '精致'],
        '粗糙': ['自然', '原始', '石头', '树皮', '质朴', '真实', '力量'],
        '轻微纹理': ['自然', '有机', '温和', '舒适', '生活'],
    }
}

# 概念关联网络 - 定义概念之间的联想关系（用于链式联想）
CONCEPT_NETWORK = {
    # 自然类
    '苹果': {'红': 0.9, '圆形': 0.85, '水果': 1.0, '甜': 0.9, '健康': 0.8, '秋天': 0.7},
    '香蕉': {'黄': 0.9, '长条形': 0.85, '水果': 1.0, '甜': 0.85, '能量': 0.8},
    '西红柿': {'红': 0.95, '圆形': 0.8, '蔬菜': 0.9, '酸甜': 0.7, '烹饪': 0.8},
    '草莓': {'红': 0.95, '小': 0.8, '水果': 1.0, '甜': 0.9, '可爱': 0.7},
    '太阳': {'圆形': 0.95, '黄': 0.9, '温暖': 1.0, '光明': 0.95, '生命': 0.85},
    '月亮': {'圆形': 0.9, '白': 0.85, '夜晚': 1.0, '浪漫': 0.8, '神秘': 0.75},
    
    # 情感类
    '热情': {'红': 0.85, '温暖': 0.9, '活力': 0.85, '爱情': 0.8},
    '温暖': {'红': 0.8, '橙': 0.85, '舒适': 0.9, '家': 0.75},
    '冷静': {'蓝': 0.9, '智慧': 0.85, '思考': 0.8},
    '快乐': {'黄': 0.85, '阳光': 0.9, '笑容': 0.85},
    
    # 抽象概念
    '生命': {'绿': 0.85, '自然': 0.9, '成长': 0.85, '希望': 0.8},
    '自然': {'绿': 0.9, '植物': 0.95, '清新': 0.85, '健康': 0.8},
    '神秘': {'黑': 0.8, '紫': 0.85, '夜晚': 0.9, '未知': 0.85},
    '爱情': {'红': 0.9, '心形': 0.85, '温暖': 0.8, '甜蜜': 0.75},
    
    # 人类相关
    '人脸': {'人类': 1.0, '情感': 0.9, '交流': 0.85, '记忆': 0.8, '故事': 0.75},
    '人类': {'生命': 0.9, '情感': 0.85, '社会': 0.8, '智慧': 0.75},
    '儿童': {'小': 0.7, '可爱': 0.9, '纯真': 0.85, '未来': 0.8},
    '青年': {'活力': 0.85, '希望': 0.8, '成长': 0.75},
    '老年': {'智慧': 0.9, '经验': 0.85, '平静': 0.8},
    
    # 电子设备相关
    '手机': {'长方形': 0.9, '屏幕': 0.8, '手掌': 0.7, '人脸': 0.6},
    '手持物体': {'人脸': 0.7, '长方形': 0.8},
}

# 经验关联类型
RELATION_TYPES = {
    'similar': '相似',      # 视觉特征相似
    'related': '相关',      # 功能或用途相关
    'opposite': '对立',     # 特征对立
    'part_of': '部分',      # 整体-部分关系
    'caused_by': '因果',    # 因果关系
    'associated': '联想',   # 概念联想
    'sequential': '顺序',   # 时间或逻辑顺序
}

def _log(level: str, message: str):
    """日志输出"""
    if DEBUG_MODE:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # 使用更明显的分隔符
        if '==========' in message:
            print(f"\n[{timestamp}] [INFO] {message}")
        elif '[联想推理]' in message:
            print(f"[{timestamp}] [{level}] {message}")
        else:
            print(f"[{timestamp}] [{level}] [联想推理] {message}")


class AssociativeMemory:
    """联想记忆模块 - 实现形象思维的联想推理"""

    def __init__(self, memory_instance, relations_file: str = None):
        """
        初始化联想记忆模块

        Args:
            memory_instance: ExperienceMemory 实例
            relations_file: 经验关联关系文件路径
        """
        self.memory = memory_instance
        self.relations_file = relations_file or "experience_relations.json"
        self.relations = {}  # 经验关联关系 {exp_id: [(related_exp_id, relation_type, strength)]}
        self.feature_index = {}  # 特征索引 {feature_type: {feature_value: [exp_indices]}}
        
        self._build_feature_index()
        self._load_relations()
        
        _log("INFO", "联想记忆模块初始化完成")
    
    def _build_feature_index(self):
        """构建特征索引，加速联想查询"""
        self.feature_index = {
            '颜色': {},
            '形状': {},
            '大小': {},
            '纹理': {},
        }
        
        for idx, exp in enumerate(self.memory.experiences):
            cond = exp.get('condition', {})
            for feature_type in self.feature_index.keys():
                value = cond.get(feature_type)
                if value:
                    if value not in self.feature_index[feature_type]:
                        self.feature_index[feature_type][value] = []
                    self.feature_index[feature_type][value].append(idx)
        
        _log("DEBUG", f"特征索引构建完成: {[(k, len(v)) for k, v in self.feature_index.items()]}")
    
    def _load_relations(self):
        """加载经验关联关系"""
        if os.path.exists(self.relations_file):
            try:
                with open(self.relations_file, 'r', encoding='utf-8') as f:
                    self.relations = json.load(f)
                _log("INFO", f"加载 {len(self.relations)} 条经验关联关系")
            except Exception as e:
                _log("WARNING", f"加载关联关系失败: {e}")
                self.relations = {}
    
    def _save_relations(self):
        """保存经验关联关系"""
        try:
            with open(self.relations_file, 'w', encoding='utf-8') as f:
                json.dump(self.relations, f, indent=2, ensure_ascii=False)
            _log("DEBUG", f"保存关联关系到 {self.relations_file}")
        except Exception as e:
            _log("ERROR", f"保存关联关系失败: {e}")
    
    def get_feature_associations(self, feature_type: str, feature_value: str) -> List[str]:
        """
        获取特征的联想词
        
        Args:
            feature_type: 特征类型（颜色、形状等）
            feature_value: 特征值（红、圆形等）
        
        Returns:
            联想词列表
        """
        associations = FEATURE_ASSOCIATIONS.get(feature_type, {}).get(feature_value, [])
        return associations
    
    def find_similar_experiences(self, marks_dict: Dict, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """
        基于特征联想找到相似的经验
        
        Args:
            marks_dict: 当前特征
            top_k: 返回前k个最相似的经验
        
        Returns:
            [(经验索引, 相似度, 联想说明), ...]
        """
        _log("INFO", "========== 联想推理开始 ==========")
        _log("DEBUG", f"输入特征: {marks_dict}")
        
        results = []
        
        for idx, exp in enumerate(self.memory.experiences):
            cond = exp.get('condition', {})
            similarity = 0.0
            associations = []
            
            # 颜色联想
            color = marks_dict.get('颜色')
            exp_color = cond.get('颜色')
            if color and exp_color:
                if color == exp_color:
                    similarity += 0.35
                    associations.append(f"颜色相同({color})")
                elif self._are_colors_related(color, exp_color):
                    similarity += 0.2
                    associations.append(f"颜色相近({color}→{exp_color})")
            
            # 形状联想
            shape = marks_dict.get('形状')
            exp_shape = cond.get('形状')
            if shape and exp_shape:
                if shape == exp_shape:
                    similarity += 0.35
                    associations.append(f"形状相同({shape})")
                elif self._are_shapes_related(shape, exp_shape):
                    similarity += 0.25
                    associations.append(f"形状相似({shape}→{exp_shape})")
            
            # 大小联想
            size = marks_dict.get('大小')
            exp_size = cond.get('大小')
            if size and exp_size:
                if size == exp_size:
                    similarity += 0.15
                    associations.append(f"大小相同({size})")
            
            # 纹理联想
            texture = marks_dict.get('纹理')
            exp_texture = cond.get('纹理')
            if texture and exp_texture:
                if texture == exp_texture:
                    similarity += 0.1
                    associations.append(f"纹理相同({texture})")
            
            # 人脸特征联想
            if shape == '人脸' and exp_shape == '人脸':
                age = marks_dict.get('年龄')
                exp_age = cond.get('年龄')
                gender = marks_dict.get('性别')
                exp_gender = cond.get('性别')
                
                if age and exp_age and age == exp_age:
                    similarity += 0.15
                    associations.append(f"年龄相同({age})")
                if gender and exp_gender and gender == exp_gender:
                    similarity += 0.15
                    associations.append(f"性别相同({gender})")
            
            if similarity > 0.1:
                results.append((idx, similarity, {
                    'exp': exp,
                    'associations': associations,
                    'name': exp.get('name', '未知')
                }))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        _log("INFO", f"找到 {len(results)} 个相关经验")
        for i, (idx, sim, info) in enumerate(results[:top_k]):
            _log("DEBUG", f"  #{i+1}: {info['name']} (相似度: {sim:.2f}, 联想: {', '.join(info['associations'])})")
        
        return results[:top_k]
    
    def _are_colors_related(self, color1: str, color2: str) -> bool:
        """判断两个颜色是否相关"""
        color_groups = [
            {'红', '橙', '粉'},
            {'黄', '金', '橙'},
            {'蓝', '青', '紫'},
            {'绿', '青'},
            {'黑', '灰', '白'},
        ]
        for group in color_groups:
            if color1 in group and color2 in group:
                return True
        return False
    
    def _are_shapes_related(self, shape1: str, shape2: str) -> bool:
        """判断两个形状是否相关"""
        from rightbrain.learning.memory import SHAPE_SIMILARITY
        if shape1 in SHAPE_SIMILARITY:
            return shape2 in SHAPE_SIMILARITY[shape1]
        if shape2 in SHAPE_SIMILARITY:
            return shape1 in SHAPE_SIMILARITY[shape2]
        return False
    
    def create_association(self, exp_id1: int, exp_id2: int, relation_type: str, strength: float = 0.5):
        """
        创建经验之间的关联
        
        Args:
            exp_id1: 经验1索引
            exp_id2: 经验2索引
            relation_type: 关联类型
            strength: 关联强度 (0-1)
        """
        if str(exp_id1) not in self.relations:
            self.relations[str(exp_id1)] = []
        
        # 检查是否已存在
        for rel in self.relations[str(exp_id1)]:
            if rel[0] == exp_id2:
                # 更新强度
                rel[2] = max(rel[2], strength)
                self._save_relations()
                return
        
        self.relations[str(exp_id1)].append([exp_id2, relation_type, strength])
        self._save_relations()
        _log("DEBUG", f"创建关联: 经验#{exp_id1} --[{relation_type}]--> 经验#{exp_id2} (强度: {strength})")
    
    def get_related_experiences(self, exp_id: int) -> List[Tuple[int, str, float]]:
        """
        获取与指定经验相关的所有经验
        
        Args:
            exp_id: 经验索引
        
        Returns:
            [(相关经验索引, 关联类型, 关联强度), ...]
        """
        return self.relations.get(str(exp_id), [])
    
    def analogical_reasoning(self, marks_dict: Dict) -> Dict:
        """
        类比推理 - 基于相似经验进行推理
        
        Args:
            marks_dict: 当前特征
        
        Returns:
            推理结果，包含类比对象和推理结论
        """
        _log("INFO", "========== 类比推理开始 ==========")
        
        # 找到相似经验
        similar_exps = self.find_similar_experiences(marks_dict, top_k=3)
        
        if not similar_exps:
            _log("INFO", "未找到相似经验，无法进行类比推理")
            return {
                'success': False,
                'reason': '未找到相似经验',
                'suggestion': '需要学习新经验'
            }
        
        # 获取最佳匹配
        best_idx, best_sim, best_info = similar_exps[0]
        best_exp = best_info['exp']
        
        # 获取联想词
        associations = []
        for feature_type in ['颜色', '形状', '大小', '纹理']:
            value = marks_dict.get(feature_type)
            if value:
                feature_assocs = self.get_feature_associations(feature_type, value)
                associations.extend(feature_assocs)
        
        # 注意：手机识别已移至深度传感器（YOLO），避免重复逻辑
        
        # 构建推理结果
        result = {
            'success': True,
            'analogical_object': best_exp.get('name', '未知'),
            'similarity': best_sim,
            'associations': list(set(associations))[:5],  # 去重，最多5个
            'reasoning': f"这个物体与{best_exp.get('name', '未知')}相似（{', '.join(best_info['associations'])}）",
            'suggested_action': best_exp.get('action', '未知'),
            'confidence': best_sim * best_exp.get('confidence', 0.5),
        }
        
        _log("INFO", f"类比推理结果: {result['analogical_object']} (置信度: {result['confidence']:.2f})")
        
        return result
    
    def reinforce_experience(self, exp_id: int, feedback: str = 'positive'):
        """
        强化经验 - 通过反馈调整置信度
        
        Args:
            exp_id: 经验索引
            feedback: 反馈类型 ('positive' 或 'negative')
        """
        if exp_id < 0 or exp_id >= len(self.memory.experiences):
            _log("WARNING", f"无效的经验索引: {exp_id}")
            return
        
        exp = self.memory.experiences[exp_id]
        current_confidence = exp.get('confidence', 0.5)
        
        if feedback == 'positive':
            # 正反馈：增加置信度
            new_confidence = min(1.0, current_confidence + 0.1)
            _log("INFO", f"正反馈强化: {exp.get('name', '未知')} 置信度 {current_confidence:.2f} -> {new_confidence:.2f}")
        else:
            # 负反馈：降低置信度
            new_confidence = max(0.1, current_confidence - 0.15)
            _log("INFO", f"负反馈调整: {exp.get('name', '未知')} 置信度 {current_confidence:.2f} -> {new_confidence:.2f}")
        
        exp['confidence'] = new_confidence
        exp['last_reinforced'] = time.time()
        
        # 保存到文件
        if self.memory.json_path:
            self.memory.save(self.memory.json_path)
    
    def learn_with_associations(self, new_exp: Dict) -> Dict:
        """
        学习新经验并自动建立关联
        
        Args:
            new_exp: 新经验
        
        Returns:
            学习结果，包含建立的关联
        """
        _log("INFO", "========== 关联学习开始 ==========")
        
        # 添加到经验库
        is_new = self.memory.add_or_update(new_exp)
        new_idx = len(self.memory.experiences) - 1
        
        # 重建特征索引
        self._build_feature_index()
        
        # 自动建立关联
        associations_created = []
        new_cond = new_exp.get('condition', {})
        
        for idx, exp in enumerate(self.memory.experiences):
            if idx == new_idx:
                continue
            
            cond = exp.get('condition', {})
            relation_type = None
            strength = 0.0
            
            # 检查相似关系
            common_features = 0
            for feature in ['颜色', '形状', '大小', '纹理']:
                if new_cond.get(feature) and cond.get(feature) and new_cond.get(feature) == cond.get(feature):
                    common_features += 1
            
            if common_features >= 2:
                relation_type = 'similar'
                strength = common_features * 0.25
            elif new_cond.get('颜色') == cond.get('颜色') and new_cond.get('颜色'):
                relation_type = 'similar'
                strength = 0.2
            
            if relation_type:
                self.create_association(new_idx, idx, relation_type, strength)
                associations_created.append({
                    'related_exp': exp.get('name', '未知'),
                    'type': relation_type,
                    'strength': strength
                })
        
        _log("INFO", f"学习完成，建立 {len(associations_created)} 个关联")
        
        return {
            'is_new': is_new,
            'exp_index': new_idx,
            'associations_created': associations_created
        }
    
    def get_memory_network_stats(self) -> Dict:
        """
        获取记忆网络统计信息
        
        Returns:
            统计信息
        """
        total_experiences = len(self.memory.experiences)
        total_relations = sum(len(rels) for rels in self.relations.values())
        
        # 计算平均置信度
        confidences = [exp.get('confidence', 0.5) for exp in self.memory.experiences]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # 计算特征分布
        feature_dist = {}
        for feature_type in self.feature_index:
            feature_dist[feature_type] = len(self.feature_index[feature_type])
        
        return {
            'total_experiences': total_experiences,
            'total_relations': total_relations,
            'average_confidence': avg_confidence,
            'feature_distribution': feature_dist,
            'most_connected': self._get_most_connected_experiences(3)
        }
    
    def _get_most_connected_experiences(self, top_k: int = 5) -> List[Dict]:
        """获取关联最多的经验"""
        connection_counts = []
        for idx, exp in enumerate(self.memory.experiences):
            count = len(self.relations.get(str(idx), []))
            if count > 0:
                connection_counts.append({
                    'name': exp.get('name', '未知'),
                    'index': idx,
                    'connections': count
                })
        
        connection_counts.sort(key=lambda x: x['connections'], reverse=True)
        return connection_counts[:top_k]
    
    def chain_association(self, start_concept: str, max_depth: int = 3, min_strength: float = 0.3) -> Dict:
        """
        链式联想 - 从一个概念出发，进行多层次的联想推理
        
        Args:
            start_concept: 起始概念（可以是特征值或经验名称）
            max_depth: 最大联想深度
            min_strength: 最小关联强度
        
        Returns:
            联想链条结果
        """
        _log("INFO", f"========== 链式联想开始 ==========")
        _log("INFO", f"[输入] 起始概念: '{start_concept}', 最大深度: {max_depth}, 最小强度: {min_strength}")
        
        chain = []
        visited = set()
        
        def _expand(concept: str, depth: int, strength: float):
            if depth > max_depth or concept in visited:
                return
            
            visited.add(concept)
            
            # 查找概念的所有关联
            associations = []
            source_info = []
            
            # 1. 从特征关联中查找
            _log("DEBUG", f"[深度{depth}] 从特征映射中查找 '{concept}'")
            for feature_type, feature_map in FEATURE_ASSOCIATIONS.items():
                if concept in feature_map:
                    for assoc in feature_map[concept]:
                        associations.append((assoc, 'feature', 0.7))
                        source_info.append(f"特征映射[{feature_type}]")
                    _log("DEBUG", f"  -> 在 {feature_type} 映射中找到: {[a[0] for a in feature_map[concept]]}")
            
            # 2. 从概念网络中查找
            _log("DEBUG", f"[深度{depth}] 从概念网络中查找 '{concept}'")
            if concept in CONCEPT_NETWORK:
                for assoc, str_val in CONCEPT_NETWORK[concept].items():
                    if str_val >= min_strength:
                        associations.append((assoc, 'concept', str_val))
                        source_info.append(f"概念网络")
                _log("DEBUG", f"  -> 在概念网络中找到: {[a[0] for a in associations if a[1] == 'concept']}")
            
            # 3. 从经验库中查找
            _log("DEBUG", f"[深度{depth}] 从经验库中查找 '{concept}'")
            for idx, exp in enumerate(self.memory.experiences):
                if exp.get('name') == concept:
                    exp_associations = []
                    # 查找该经验的所有关联经验
                    for rel in self.relations.get(str(idx), []):
                        rel_exp = self.memory.experiences[rel[0]]
                        exp_associations.append((rel_exp.get('name', '未知'), rel[1], rel[2]))
                        associations.append((rel_exp.get('name', '未知'), rel[1], rel[2]))
                        source_info.append(f"经验库[{exp.get('name')}]")
                    if exp_associations:
                        _log("DEBUG", f"  -> 经验 '{exp.get('name')}' 关联到: {[a[0] for a in exp_associations]}")
            
            # 去重并按强度排序
            seen = set()
            unique_assocs = []
            for assoc in associations:
                if assoc[0] not in seen:
                    seen.add(assoc[0])
                    unique_assocs.append(assoc)
            associations = sorted(unique_assocs, key=lambda x: x[2], reverse=True)
            
            # 添加到链条
            if associations:
                chain_step = {
                    'depth': depth,
                    'from': concept,
                    'associations': associations[:5],  # 最多5个关联
                    'strength': strength
                }
                chain.append(chain_step)
                
                top_assocs = [f"{a[0]}({a[2]:.2f})" for a in associations[:3]]
                _log("INFO", f"[深度{depth}] ✓ {concept} -> [{', '.join(top_assocs)}]")
                
                # 递归扩展（选择最强的关联继续）
                for assoc_name, assoc_type, assoc_strength in associations[:2]:
                    if assoc_strength >= min_strength and assoc_name not in visited:
                        _log("DEBUG", f"[递归] 继续扩展: '{concept}' -> '{assoc_name}' (强度: {assoc_strength})")
                        _expand(assoc_name, depth + 1, assoc_strength * strength)
        
        _expand(start_concept, 1, 1.0)
        
        # 构建思维链条描述
        chain_description = self._build_chain_description(chain)
        
        _log("INFO", f"[完成] 链式联想完成，共 {len(chain)} 层，总关联数: {sum(len(step['associations']) for step in chain)}")
        if chain:
            chain_path = " → ".join([step['from'] for step in chain])
            _log("INFO", f"[路径] {chain_path}")
        
        return {
            'success': len(chain) > 0,
            'start_concept': start_concept,
            'chain': chain,
            'description': chain_description,
            'total_associations': sum(len(step['associations']) for step in chain)
        }
    
    def _build_chain_description(self, chain: List[Dict]) -> str:
        """构建思维链条的自然语言描述"""
        if not chain:
            return "没有找到关联"
        
        descriptions = []
        for step in chain:
            from_concept = step['from']
            top_assocs = [a[0] for a in step['associations'][:3]]
            descriptions.append(f"{from_concept}让我联想到{'、'.join(top_assocs)}")
        
        return " → ".join([step['from'] for step in chain]) + f"\n\n思维过程：{'; '.join(descriptions)}"
    
    def imaginal_thinking(self, marks_dict: Dict) -> Dict:
        """
        形象思维 - 综合特征进行形象化的联想推理
        
        Args:
            marks_dict: 当前特征
        
        Returns:
            形象思维结果，包含联想链条和形象描述
        """
        _log("INFO", "="*50)
        _log("INFO", "========== 形象思维开始 ==========")
        _log("INFO", f"[输入] 物体特征: {marks_dict}")
        
        # 1. 提取所有特征
        features = []
        for feature_type in ['颜色', '形状', '大小', '纹理']:
            value = marks_dict.get(feature_type)
            if value and value != '未知':
                features.append((feature_type, value))
        
        _log("INFO", f"[步骤1] 提取有效特征: {features}")
        
        if not features:
            _log("WARNING", "[失败] 没有可用的特征进行形象思维")
            return {
                'success': False,
                'reason': '没有可用的特征进行形象思维'
            }
        
        # 2. 从每个特征出发进行联想
        _log("INFO", f"[步骤2] 从 {len(features)} 个特征出发进行联想")
        all_associations = {}
        for feature_type, value in features:
            assocs = self.get_feature_associations(feature_type, value)
            _log("DEBUG", f"  - {feature_type}='{value}' -> {assocs[:5]}")
            for assoc in assocs:
                if assoc not in all_associations:
                    all_associations[assoc] = []
                all_associations[assoc].append((feature_type, value))
        
        _log("INFO", f"[步骤2] 联想结果: 共 {len(all_associations)} 个概念")
        
        # 3. 找出被多个特征共同联想的概念（交叉点）
        _log("INFO", "[步骤3] 寻找交叉点（被多个特征共同联想的概念）")
        crossroads = []
        for assoc, sources in all_associations.items():
            if len(sources) > 1:
                crossroads.append({
                    'concept': assoc,
                    'sources': sources,
                    'strength': len(sources) / len(features)
                })
                _log("DEBUG", f"  ★ 交叉点: '{assoc}' (被 {len(sources)} 个特征联想: {[s[0] for s in sources]})")
        
        crossroads.sort(key=lambda x: x['strength'], reverse=True)
        _log("INFO", f"[步骤3] 发现 {len(crossroads)} 个交叉点")
        
        if crossroads:
            top_cross = crossroads[0]
            _log("INFO", f"[最优] 首选交叉点: '{top_cross['concept']}' (强度: {top_cross['strength']:.2f})")
        
        # 4. 从交叉点出发进行链式联想
        _log("INFO", f"[步骤4] 从交叉点出发进行链式联想")
        chain_results = []
        for i, cr in enumerate(crossroads[:3]):
            _log("DEBUG", f"  - 交叉点{i+1}: '{cr['concept']}'")
            chain = self.chain_association(cr['concept'], max_depth=2)
            if chain['success']:
                chain_results.append(chain)
                _log("DEBUG", f"    -> 链式联想成功，{len(chain['chain'])} 层")
        
        _log("INFO", f"[步骤4] 链式联想完成: {len(chain_results)} 条链条")
        
        # 5. 构建形象描述
        image_description = self._build_image_description(marks_dict, crossroads, chain_results)
        
        _log("INFO", f"[完成] 形象思维完成")
        _log("INFO", f"  - 发现交叉点: {len(crossroads)} 个")
        _log("INFO", f"  - 链式联想: {len(chain_results)} 条")
        _log("INFO", f"  - 形象描述: {image_description[:50]}...")
        
        return {
            'success': True,
            'features': features,
            'crossroads': crossroads[:5],
            'chains': chain_results,
            'image_description': image_description,
            'all_associations': dict(list(all_associations.items())[:10])
        }
    
    def _build_image_description(self, marks_dict: Dict, crossroads: List[Dict], chains: List[Dict]) -> str:
        """构建形象化的描述"""
        color = marks_dict.get('颜色', '未知')
        shape = marks_dict.get('形状', '未知')
        size = marks_dict.get('大小', '未知')
        
        # 基础描述
        # 形状名称已经包含了'形'字（如'长方形''圆形'），不需要再追加
        shape_desc = shape
        if not shape.endswith('形'):
            shape_desc = f"{shape}形"
        base_desc = f"这是一个{color}色的{shape_desc}物体，尺寸{size}。"
        
        # 联想描述
        if crossroads:
            top_cross = crossroads[0]
            cross_desc = f"它让我联想到「{top_cross['concept']}」，因为它的{'和'.join([s[0] for s in top_cross['sources']])}特征都指向这个概念。"
        else:
            cross_desc = ""
        
        # 链式联想描述
        if chains:
            chain = chains[0]
            chain_desc = f"进一步联想：{chain['description']}"
        else:
            chain_desc = ""
        
        return f"{base_desc}\n{cross_desc}\n{chain_desc}"
    
    def generate_story(self, marks_dict: Dict) -> str:
        """
        根据特征生成一个小故事（形象化表达）
        
        Args:
            marks_dict: 当前特征
        
        Returns:
            生成的故事
        """
        # 进行形象思维
        result = self.imaginal_thinking(marks_dict)
        
        if not result['success']:
            return "我看到了一个物体，但还没有足够的联想来讲述它的故事。"
        
        color = marks_dict.get('颜色', '未知')
        shape = marks_dict.get('形状', '未知')
        
        # 构建故事
        story_parts = []
        
        # 开头
        story_parts.append(f"在一个{color}色的世界里，有一个{shape}形的存在。")
        
        # 联想部分
        crossroads = result.get('crossroads', [])
        if crossroads:
            concepts = [cr['concept'] for cr in crossroads[:3]]
            story_parts.append(f"它让人想起{'、'.join(concepts)}。")
        
        # 链式联想部分
        chains = result.get('chains', [])
        if chains:
            chain = chains[0]
            chain_concepts = [step['from'] for step in chain.get('chain', [])]
            if len(chain_concepts) > 1:
                story_parts.append(f"从{chain_concepts[0]}到{chain_concepts[-1]}，这是一段奇妙的联想之旅。")
        
        # 结尾
        story_parts.append(f"这个{color}色的{shape}形物体，承载着丰富的意义和联想。")
        
        return "\n".join(story_parts)


# 全局实例
_associative_memory = None

def get_associative_memory(memory_instance=None) -> AssociativeMemory:
    """
    获取联想记忆模块实例
    
    Args:
        memory_instance: ExperienceMemory 实例（首次调用时需要）
    
    Returns:
        AssociativeMemory 实例
    """
    global _associative_memory
    
    if _associative_memory is None and memory_instance is not None:
        _associative_memory = AssociativeMemory(memory_instance)
    
    return _associative_memory
