import cv2
import numpy as np

def get_focus_roi(image):
    """检测图像中主要物体的ROI区域，作为注意力焦点"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    
    _, thresh_binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    _, thresh_inv = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)
    
    contours_binary, _ = cv2.findContours(thresh_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_inv, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    area_binary = max([cv2.contourArea(c) for c in contours_binary]) if contours_binary else 0
    area_inv = max([cv2.contourArea(c) for c in contours_inv]) if contours_inv else 0
    
    if area_binary > area_inv and contours_binary:
        c = max(contours_binary, key=cv2.contourArea)
    elif contours_inv:
        c = max(contours_inv, key=cv2.contourArea)
    else:
        h, w = image.shape[:2]
        return (w//4, h//4, 3*w//4, 3*h//4)
    
    x, y, w, h = cv2.boundingRect(c)
    return (x, y, x+w, y+h)

def refine_focus(image, bbox, original_marks):
    """对已检测到的区域进行二次焦点精修"""
    from rightbrain.cv_core.sensor import extract_marks
    fine_marks = extract_marks(image, bbox, use_attention=False)
    merged = original_marks.copy()
    merged.update(fine_marks)
    return merged

def find_main_object_bbox(image):
    """已弃用 - 使用 get_focus_roi 替代"""
    import warnings
    warnings.warn("find_main_object_bbox is deprecated, use get_focus_roi instead", DeprecationWarning)
    return get_focus_roi(image)