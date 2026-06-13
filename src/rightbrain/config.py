import os
import logging
from typing import Dict, Any

class Config:
    DEBUG_MODE: bool = True
    LOG_LEVEL: str = "DEBUG"

    PERF_LOG_ENABLED: bool = True
    PERF_LOG_LEVEL: str = "INFO"

    # ===== 特征权重(经验匹配用) =====
    FEATURE_WEIGHTS: Dict[str, float] = {
        '颜色': 0.35,
        '形状': 0.35,
        '大小': 0.12,
        '纹理': 0.08,
        '距离': 0.05,
        '边缘距离': 0.03,
        '支撑面': 0.02,
        '年龄': 0.15,    # 人脸年龄特征权重
        '性别': 0.15,    # 人脸性别特征权重
    }
    FORGET_THRESHOLD_DAYS: int = 7
    FORGET_DECAY_RATE: float = 0.1
    WEIGHT_ADJUSTMENT_RATE: float = 0.05
    MIN_WEIGHT: float = 0.01
    MAX_WEIGHT: float = 0.5

    # YOLO 类别到中文名称的映射(统一使用)
    YOLO_CLASS_MAPPING: Dict[str, str] = {
        'cup': '水杯',
        'mug': '水杯',
        'cell phone': '手机',
        'phone': '手机',
        'mobile phone': '手机',
        'bottle': '水瓶',
        'person': '人',
        'laptop': '笔记本电脑',
        'keyboard': '键盘',
        'mouse': '鼠标',
        'book': '书',
        'chair': '椅子',
        'table': '桌子',
        'tv': '电视',
        'monitor': '显示器',
        'plant': '植物',
        'flower': '花',
        'bag': '包',
        'backpack': '背包',
        'umbrella': '雨伞',
        'hat': '帽子',
        'glasses': '眼镜',
        'watch': '手表',
        'clock': '时钟',
        'car': '汽车',
        'bicycle': '自行车',
        'motorcycle': '摩托车',
        'bus': '公交车',
        'truck': '卡车',
        'bird': '鸟',
        'cat': '猫',
        'dog': '狗',
        'apple': '苹果',
        'banana': '香蕉',
        'orange': '橙子',
        'cake': '蛋糕',
        'pizza': '披萨',
        'fork': '叉子',
        'knife': '刀',
        'spoon': '勺子',
        'bowl': '碗',
        'plate': '盘子',
        'glass': '玻璃杯',
        'wine glass': '酒杯',
        'remote': '遥控器',
        'remote control': '遥控器',
    }

    # ===== 大模型配置 =====
    OLLAMA_BASE: str = "http://127.0.0.1:11434"
    MODEL_NAME: str = "qwen3.5:9b"
    USE_ONLINE_MODEL: bool = True
    # 火山方舟配置
    VOLCANO_API_KEY: str = ""
    VOLCANO_MODEL: str = "deepseek-v4-pro-260425"
    # DeepSeek 配置
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1/chat/completions"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    REQUEST_TIMEOUT: int = 120
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 2.0

    CONFIDENCE_THRESHOLD: float = 0.6
    CONFIDENCE_DELTA: float = 0.1

    # ===== 阈值配置(基因范围钳制) =====
    CURIOUSITY_THRESHOLD: float = 0.6      # 好奇心触发阈值
    CURIOUSITY_MIN_THRESHOLD: float = 0.4  # 基因允许的下限
    WTA_MARGIN: float = 0.03               # WTA竞争胜出阈值
    SHAPE_CORRECTION_ASPECT: float = 1.5   # 三角形→长方形修正阈值
    MAX_LEARN_PER_DAY: int = 10            # 主动学习每日上限

    # 学习冷却配置
    LEARNING_COOLDOWN_SECONDS: float = 30.0  # 同一物体学习冷却时间(秒)
    FACE_AGE_CACHE_TTL: float = 2.0  # 人脸年龄缓存有效期(秒)

    SHAPE_MIN_AREA: int = 5
    SHAPE_SIMILARITY: Dict[str, list] = {
        '圆形': ['圆形', '椭圆形', '球形'],
        '椭圆形': ['圆形', '椭圆形'],
        '球形': ['圆形', '球形'],
        '方形': ['方形', '长方形', '正方形', '矩形'],
        '长方形': ['方形', '长方形', '正方形', '矩形', '长条形'],
        '正方形': ['方形', '长方形', '正方形', '矩形'],
        '矩形': ['方形', '长方形', '正方形', '矩形', '长条形'],
        '长条形': ['长条形', '长方形', '矩形'],
        '三角形': ['三角形', '三角'],
        '三角': ['三角形', '三角'],
        '多边形': ['多边形', '五边形', '六边形', '八边形', '不规则'],
        '五边形': ['五边形', '多边形'],
        '六边形': ['六边形', '多边形'],
        '八边形': ['八边形', '多边形'],
        '不规则': ['不规则', '多边形'],
    }

    SHAPE_MIN_AREA_RATIO: float = 0.01
    SHAPE_MAX_AREA_RATIO: float = 0.995
    CIRCULARITY_THRESHOLD: float = 0.55
    ELLIPSE_THRESHOLD: float = 0.5

    AFFECT_ENABLED: bool = True

    ENABLE_ILLUMINATION_CORRECTION: bool = False

    @classmethod
    def load_from_env(cls):
        # 尝试从项目根目录的 .env 文件加载
        _dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env')
        _dotenv_path = os.path.abspath(_dotenv_path)
        if os.path.exists(_dotenv_path):
            with open(_dotenv_path, 'r') as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line and not _line.startswith('#') and '=' in _line:
                        _k, _v = _line.split('=', 1)
                        _k, _v = _k.strip(), _v.strip()
                        if _k not in os.environ:  # 环境变量优先
                            os.environ[_k] = _v
        env_vars: Dict[str, Any] = {
            "DEBUG_MODE": bool,
            "LOG_LEVEL": str,
            "PERF_LOG_ENABLED": bool,
            "OLLAMA_BASE": str,
            "MODEL_NAME": str,
            "USE_ONLINE_MODEL": bool,
            "VOLCANO_API_KEY": str,
            "VOLCANO_MODEL": str,
            "DEEPSEEK_API_KEY": str,
            "DEEPSEEK_API_URL": str,
            "DEEPSEEK_MODEL": str,
            "REQUEST_TIMEOUT": int,
            "MAX_RETRIES": int,
            "RETRY_DELAY": float,
            "CONFIDENCE_THRESHOLD": float,
            "CONFIDENCE_DELTA": float,
            "CIRCULARITY_THRESHOLD": float,
            "ELLIPSE_THRESHOLD": float,
            "AFFECT_ENABLED": bool,
            "ENABLE_ILLUMINATION_CORRECTION": bool,
        }

        for key, cast_type in env_vars.items():
            value = os.environ.get(key)
            if value is not None:
                if cast_type == bool:
                    setattr(cls, key, value.lower() == "true")
                else:
                    try:
                        setattr(cls, key, cast_type(value))
                    except ValueError:
                        pass

    @classmethod
    def update(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
    
    @classmethod
    def reload(cls):
        cls.__init__()
        cls.load_from_env()
    
    @classmethod
    def clamp_to_genome(cls, key: str, value: float) -> float:
        """
        将参数钳制到基因允许的范围内。
        从 genome.CORE_CONSTRAINTS 读取范围，超限则钳制并告警。
        """
        try:
            from genome import CORE_CONSTRAINTS
        except ImportError:
            return value
        
        if key not in CORE_CONSTRAINTS:
            return value
        
        constraint = CORE_CONSTRAINTS[key]
        if isinstance(constraint, dict):
            min_v = constraint.get("min", float("-inf"))
            max_v = constraint.get("max", float("inf"))
            if value < min_v:
                print(f"[Config] ⚠️ {key}={value} 低于基因下限{min_v}，已钳制")
                return min_v
            if value > max_v:
                print(f"[Config] ⚠️ {key}={value} 高于基因上限{max_v}，已钳制")
                return max_v
        return value

Config.load_from_env()

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level = getattr(logging, Config.LOG_LEVEL.upper(), logging.DEBUG)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.propagate = False

    return logger