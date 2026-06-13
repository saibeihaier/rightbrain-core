"""
ROI 特征提取 + 右脑场景报告（标准 JSON）

输出格式标准化的 JSON 场景报告，供任何大模型直接解析。
"""
import cv2
import numpy as np
import base64
import time
from typing import Dict, Optional, List


def extract_roi_features(image: np.ndarray, bbox) -> Dict:
    """从 ROI 区域提取丰富的像素级特征。"""
    result = {}
    x1, y1, x2, y2 = _parse_bbox(bbox)
    roi = image[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return result
    h, w = roi.shape[:2]

    for i, name in enumerate(['R', 'G', 'B']):
        hist = cv2.calcHist([roi], [i], None, [8], [0, 256])
        hist = cv2.normalize(hist, hist).flatten().tolist()
        result.setdefault('color_hist', {})[name] = [round(v, 4) for v in hist]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_bins = np.histogram(hsv[:, :, 0].flatten(), bins=12, range=(0, 180))[0]
    h_bins = h_bins / max(h_bins.sum(), 1)
    result['hue_diversity'] = round(float((h_bins > 0.05).sum()), 2)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    result['texture_contrast'] = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx**2 + sobely**2)
    result['edge_mean'] = round(float(edge_mag.mean()), 2)
    result['edge_std'] = round(float(edge_mag.std()), 2)
    result['brightness_mean'] = round(float(gray.mean()), 1)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        peri = cv2.arcLength(largest, True)
        result['circularity'] = round(float(4 * np.pi * area / (peri * peri)), 4) if peri > 0 and area > 0 else 0.0
        result['solidity'] = round(float(area / (w * h)), 4)
        rect = cv2.minAreaRect(largest)
        result['aspect_ratio'] = round(float(max(rect[1]) / max(min(rect[1]), 1)), 2)
    else:
        result['circularity'] = result['solidity'] = 0.0
        result['aspect_ratio'] = round(w / max(h, 1), 2)

    result['roi_ratio'] = round((w * h) / (image.shape[0] * image.shape[1]), 4)
    return result


def _parse_bbox(bbox) -> tuple:
    if isinstance(bbox, dict):
        bbox = bbox.get('bbox', list(bbox.values())[:4])
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    return (0, 0, 0, 0)


def build_scene_report(marks: Dict, roi: Dict = None, safety: tuple = None,
                       chain_desc: str = None) -> str:
    """
    构建右脑场景报告（字符串格式，向后兼容）。
    新代码请用 build_scene_json()。
    """
    data = build_scene_json(marks, roi, safety, chain_desc)
    return _json_to_text(data)


def build_scene_json(marks: Dict, roi: Dict = None, safety: tuple = None,
                     chain_desc: str = None) -> Dict:
    """
    构建标准 JSON 格式的右脑场景报告。
    
    这是右脑"看到"的世界——给左脑（任何大模型）使用的结构化数据。
    
    返回格式：
    {
        "timestamp": 时间戳,
        "focus": {  # 主要注意力物体
            "name": "苹果",           # 识别名称（如有）
            "color": "红",
            "shape": "圆形",
            "size": "中",
            "texture": "光滑",
            "confidence": 0.79,
            "spatial": {"x": 0.35, "y": 0.65, "edge": "远", "zone": "中间"},
            "safety": {"danger": false, "reason": ""},
            "emotion": {"feeling": "中性", "pleasantness": 0.47},
            "features": {"circularity": 0.88, "aspect_ratio": 1.0}  # 来自ROI
        },
        "persons": [  # 检测到的人脸
            {"age": "青年", "gender": "男", "emotion": ""}
        ],
        "scene": "桌面上有一个苹果",  # 一句话场景摘要
        "associations": ["水果", "可以吃"],
        "context": "用户手中拿着：蓝色长方形",
        "left_brain": {  # 左脑猜测结果（如有）
            "guessed_name": "",
            "confidence": 0.0
        }
    }
    """
    report = {
        "timestamp": time.time(),
        "focus": {},
        "persons": [],
        "scene": "",
        "associations": [],
        "context": "",
        "left_brain": {"guessed_name": "", "confidence": 0.0},
    }

    color = marks.get('颜色', '未知')
    shape = marks.get('形状', '未知')
    size = marks.get('大小', '未知')
    texture = marks.get('纹理', '未知')

    # 焦点物体
    focus = {
        "name": marks.get('_exp_name', ''),
        "color": color,
        "shape": shape,
        "size": size,
        "texture": texture,
        "confidence": marks.get('_confidence', marks.get('score', 0.0)),
        "spatial": {
            "x": marks.get('空间_x', None),
            "y": marks.get('空间_y', None),
            "edge": marks.get('距边缘', None),
            "zone": marks.get('位置区域', None),
        },
        "safety": {"danger": False, "reason": ""},
        "emotion": {
            "feeling": marks.get('感受', ''),
            "pleasantness": marks.get('愉悦度', None),
        },
        "features": {},
    }

    # 安全
    if safety:
        focus["safety"] = {"danger": safety[0], "reason": safety[1]}

    # ROI 特征
    if roi:
        focus["features"] = {
            "circularity": roi.get('circularity', 0),
            "aspect_ratio": roi.get('aspect_ratio', 1),
            "texture_contrast": roi.get('texture_contrast', 0),
            "edge_mean": roi.get('edge_mean', 0),
            "brightness": roi.get('brightness_mean', 128),
            "solidity": roi.get('solidity', 0),
        }

    report["focus"] = focus

    # 人脸
    if marks.get('年龄') or marks.get('性别'):
        report["persons"].append({
            "age": marks.get('年龄', ''),
            "gender": marks.get('性别', ''),
            "emotion": marks.get('表情', ''),
        })

    # 联想
    if chain_desc:
        report["associations"] = [chain_desc]

    # 场景摘要
    name = focus["name"] or f"{color}{shape}"
    scene_parts = [f"看到{color}{shape}"]
    if focus["name"]:
        scene_parts = [f"看到{name}"]
    if focus["spatial"]["zone"]:
        scene_parts.append(f"在{focus['spatial']['zone']}")
    report["scene"] = "".join(scene_parts)

    return report


def _json_to_text(data: Dict) -> str:
    """把 JSON 报告转回文字描述（向后兼容）"""
    focus = data.get("focus", {})
    lines = []

    color = focus.get("color", "未知")
    shape = focus.get("shape", "未知")
    size = focus.get("size", "未知")
    name = focus.get("name", "")

    if name:
        lines.append(f"【主要物体】{color}{shape}{size}，识别为{name}（置信度{focus.get('confidence', 0):.2f}）")
    else:
        lines.append(f"【主要物体】{color}{shape}{size}")

    zone = focus.get("spatial", {}).get("zone")
    edge = focus.get("spatial", {}).get("edge")
    if zone or edge:
        parts = []
        if zone: parts.append(f"位置{zone}")
        if edge: parts.append(f"距边缘{edge}")
        lines.append(f"【空间】{'，'.join(parts)}")

    safe = focus.get("safety", {})
    if safe.get("danger"):
        lines.append(f"【安全】⚠️ {safe.get('reason', '')}")
    else:
        lines.append(f"【安全】✅ 安全")

    for p in data.get("persons", []):
        lines.append(f"【人脸】{p.get('age', '')}{p.get('gender', '')}")

    assoc = data.get("associations", [])
    if assoc:
        lines.append(f"【联想】{assoc[0]}")

    return "\n".join(lines)


def features_to_debug_string(roi_features: Dict) -> str:
    lines = [
        f"  纹理={roi_features.get('texture_contrast', 0)}",
        f"  边缘={roi_features.get('edge_mean', 0)}",
        f"  圆形度={roi_features.get('circularity', 0)}",
        f"  宽高比={roi_features.get('aspect_ratio', 0)}",
        f"  亮度={roi_features.get('brightness_mean', 0)}",
    ]
    return "\n".join(lines)
