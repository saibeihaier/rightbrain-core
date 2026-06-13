from setuptools import setup, find_packages

setup(
    name="rightbrain",
    version="1.4.0",
    description="RightBrain 视觉智能系统 - 右脑直觉 + 左脑语言混合架构",
    author="Donghua",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "llm": ["requests>=2.31.0"],
        "speech": ["vosk>=0.3.45", "sounddevice>=0.4.6", "edge-tts>=6.1.0"],
        "dl": ["torch>=2.0.0", "ultralytics>=8.0.0"],
        "all": [
            "requests>=2.31.0",
            "vosk>=0.3.45",
            "sounddevice>=0.4.6",
            "edge-tts>=6.1.0",
            "torch>=2.0.0",
            "ultralytics>=8.0.0",
            "scikit-learn>=1.3.0",
            "mediapipe>=0.10.0",
            "pygame>=2.5.0",
        ],
    },
)
