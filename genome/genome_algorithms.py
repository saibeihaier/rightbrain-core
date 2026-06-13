"""
genome/genome_algorithms.py — 基因算法实现

注意：这些算法函数现在仅供参考，与规则分离。

在新的四层架构中：
- genome_spec.json 定义规则结构（声明）
- interpreter.py 提供规则解释
- genome_algorithms.py 中的函数是可执行逻辑

AI不应该直接调用这些函数，而应该通过 interpreter 获取规则描述。
这些函数的存在只是为了向后兼容和参考实现。
"""

# 此文件保留旧版本的核心算法函数，供向后兼容和参考使用
# 实际系统使用新的四层架构

def color_detection_genome(h, s, v):
    """
    颜色识别基因：HSV 空间分段规则。

    注意：此函数仅供参考。实际颜色识别逻辑在 cv_core/sensor.py 中实现。
    """
    # 肤色
    if 0 < h < 40 and 10 < s < 120 and 40 < v < 240:
        return "肤色"
    # 低饱和度（灰色系）
    if s < 30:
        if v > 220: return "白"
        if 180 <= v <= 220: return "亮灰"
        if 120 <= v < 180: return "中灰"
        if 60 <= v < 120: return "深灰"
        if 20 <= v < 60: return "暗灰"
        return "黑"
    # 主色分段
    if (h >= 0 and h <= 10) or (h >= 160 and h <= 180):
        return "红"
    if 10 < h <= 22: return "橙"
    if 22 < h <= 55: return "黄"
    if 55 < h <= 85: return "绿"
    if 80 < h <= 90: return "青"
    if 85 < h <= 140: return "蓝"
    if 130 < h <= 160: return "紫"
    return "未知"


def shape_correction_genome(verts, aspect_ratio):
    """
    形状修正基因：三角形轮廓若长宽比 > 1.5 则修正为长方形。

    注意：此函数仅供参考。实际形状修正在 cv_core/sensor.py 中实现。
    """
    if verts != 3:
        return None, False
    if aspect_ratio > 1.5:
        return "长方形", True
    return "三角形", False


def shape_long_rect_genome(aspect_ratio):
    """
    长矩形判别基因：宽高比 > 3.0 判定为长条形。
    """
    if 0.85 <= aspect_ratio <= 1.15:
        return "正方形"
    if aspect_ratio > 3.0:
        return "长条形"
    return "长方形"


def wta_decision_genome(scores, threshold=0.03):
    """
    WTA 竞争决策基因：

    如果第一名超过第二名的比例 >= threshold，第一名胜出。
    否则不确定，返回 (-1, False)。

    注意：此函数仅供参考。实际WTA逻辑在 learning/memory2.py 中实现。
    """
    if not scores:
        return -1, False
    if len(scores) == 1:
        return 0, True
    bf = scores[0][0]
    br = scores[1][0]
    if bf <= 0:
        return -1, False
    margin = (bf - br) / bf
    if margin >= threshold:
        return 0, True
    return -1, False


def memory_match_genome(marks, condition, weights=None):
    """
    经验匹配计算基因：加权匹配度计算。

    注意：此函数仅供参考。实际匹配逻辑在 learning/memory.py 中实现。

    匹配规则：
    - 相同特征: +1.0
    - 形状相似: +0.7
    - 颜色同色系: +0.6
    - 位置宽容度: 上方≈中间 0.6，地面≈中间 0.6
    """
    if not condition:
        return 0.0

    matched = 0.0
    total = float(len(condition))

    # 形状相似组
    SHAPE_SIMILARITY = {
        '圆形': ['圆形', '椭圆形', '球形'],
        '椭圆形': ['圆形', '椭圆形'],
        '球形': ['圆形', '球形'],
        '长方形': ['长方形', '长条形', '矩形'],
        '长条形': ['长条形', '长方形', '矩形'],
        '矩形': ['长方形', '长条形', '矩形'],
    }

    # 颜色同组
    COLOR_GROUPS = {
        'warm': ['红', '橙', '黄', '粉'],
        'blueish': ['蓝', '青', '紫'],
        'greenish': ['绿'],
        'neutral': ['白', '灰', '黑', '棕'],
    }

    def _shape_similar(s1, s2):
        if s1 == s2:
            return True
        if s1 in SHAPE_SIMILARITY:
            return s2 in SHAPE_SIMILARITY[s1]
        return False

    for key, expected_val in condition.items():
        actual_val = marks.get(key)
        if actual_val is None:
            continue
        if actual_val == expected_val:
            matched += 1.0
            continue
        if key == '形状' and _shape_similar(actual_val, expected_val):
            matched += 0.7
            continue
        long_types = ['长方形', '长条形', '矩形']
        if key == '形状' and actual_val in long_types and expected_val in long_types:
            matched += 0.9
            continue
        if key == '颜色':
            for group_name, colors in COLOR_GROUPS.items():
                if actual_val in colors and expected_val in colors:
                    matched += 0.6
                    break
        if key == '位置区域':
            if expected_val == '上方' and actual_val == '中间':
                matched += 0.6
            elif expected_val == '地面' and actual_val == '中间':
                matched += 0.6

    return matched / max(total, 1.0)
