import cv2
import numpy as np
import time
import random

class MockVideoStream:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.objects = []
        self.frame_count = 0
        self._is_opened = True  # Mock流总是"打开"状态
        self._init_objects()
    
    def isOpened(self):
        """返回True表示视频源已打开"""
        return self._is_opened
    
    def _init_objects(self):
        self.objects = []
        
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (128, 0, 128)]
        shape_types = ['circle', 'square', 'rectangle', 'triangle']
        
        for i in range(3):
            obj = {
                'type': 'shape',
                'shape': random.choice(shape_types),
                'color': colors[i],
                'x': random.randint(50, self.width - 50),
                'y': random.randint(50, self.height - 50),
                'size': random.randint(30, 80),
                'dx': (random.random() - 0.5) * 2,
                'dy': (random.random() - 0.5) * 2
            }
            self.objects.append(obj)
        
        self.objects.append({
            'type': 'face',
            'x': self.width // 2 - 75,
            'y': self.height // 2 - 80,
            'size': 150,
            'dx': 0,
            'dy': 0
        })
    
    def _draw_face(self, frame, x, y, size):
        center_x, center_y = x + size // 2, y + size // 2
        radius = size // 2
        
        skin_color = (92, 148, 198)
        dark_skin = (70, 120, 170)
        light_skin = (110, 170, 220)
        
        for r in range(radius, radius // 2, -2):
            alpha = 1 - (r / radius)
            b = int(skin_color[0] * (1 - alpha) + light_skin[0] * alpha)
            g = int(skin_color[1] * (1 - alpha) + light_skin[1] * alpha)
            r_color = int(skin_color[2] * (1 - alpha) + light_skin[2] * alpha)
            cv2.circle(frame, (center_x, center_y), r, (b, g, r_color), -1)
        
        hair_color = (40, 30, 20)
        hair_radius = int(radius * 0.8)
        cv2.ellipse(frame, (center_x, center_y - radius // 4), (hair_radius, int(hair_radius * 0.4)), 0, 180, 360, hair_color, -1)
        
        eye_radius = radius // 6
        eye_offset_x = radius // 3
        eye_offset_y = radius // 5
        
        white = (255, 255, 255)
        black = (0, 0, 0)
        brown = (50, 40, 30)
        
        cv2.circle(frame, (center_x - eye_offset_x, center_y - eye_offset_y), eye_radius + 3, white, -1)
        cv2.circle(frame, (center_x + eye_offset_x, center_y - eye_offset_y), eye_radius + 3, white, -1)
        
        cv2.circle(frame, (center_x - eye_offset_x, center_y - eye_offset_y), eye_radius, brown, -1)
        cv2.circle(frame, (center_x + eye_offset_x, center_y - eye_offset_y), eye_radius, brown, -1)
        
        cv2.circle(frame, (center_x - eye_offset_x + 1, center_y - eye_offset_y - 1), eye_radius // 3, black, -1)
        cv2.circle(frame, (center_x + eye_offset_x + 1, center_y - eye_offset_y - 1), eye_radius // 3, black, -1)
        
        cv2.circle(frame, (center_x - eye_offset_x + 2, center_y - eye_offset_y - 2), eye_radius // 6, white, -1)
        cv2.circle(frame, (center_x + eye_offset_x + 2, center_y - eye_offset_y - 2), eye_radius // 6, white, -1)
        
        eyebrow_length = radius // 2
        eyebrow_height = radius // 15
        eyebrow_offset_y = center_y - eye_offset_y - radius // 3
        
        cv2.ellipse(frame, (center_x - eye_offset_x, eyebrow_offset_y), (eyebrow_length // 2, eyebrow_height), 0, 10, 170, hair_color, 2)
        cv2.ellipse(frame, (center_x + eye_offset_x, eyebrow_offset_y), (eyebrow_length // 2, eyebrow_height), 0, 10, 170, hair_color, 2)
        
        nose_width = radius // 4
        nose_height = radius // 3
        nose_points = np.array([[center_x, center_y - nose_height // 3],
                               [center_x - nose_width // 2, center_y + nose_height // 2],
                               [center_x + nose_width // 2, center_y + nose_height // 2]], np.int32)
        cv2.fillPoly(frame, [nose_points], dark_skin)
        
        mouth_width = radius // 2
        mouth_height = radius // 5
        mouth_y = center_y + radius // 3
        
        cv2.ellipse(frame, (center_x, mouth_y), (mouth_width, mouth_height), 0, 0, 180, (80, 20, 20), 3)
        
        cheek_color = (120, 160, 200)
        cheek_radius = radius // 4
        cv2.circle(frame, (center_x - radius // 2, center_y + radius // 6), cheek_radius, cheek_color, -1)
        cv2.circle(frame, (center_x + radius // 2, center_y + radius // 6), cheek_radius, cheek_color, -1)
    
    def _draw_shape(self, frame, shape, color, x, y, size):
        if shape == 'circle':
            cv2.circle(frame, (x, y), size // 2, color, -1)
        elif shape == 'square':
            cv2.rectangle(frame, (x - size // 2, y - size // 2), 
                         (x + size // 2, y + size // 2), color, -1)
        elif shape == 'rectangle':
            cv2.rectangle(frame, (x - size // 2, y - size // 3), 
                         (x + size // 2, y + size // 3), color, -1)
        elif shape == 'triangle':
            pts = np.array([[x, y - size // 2], 
                           [x - size // 2, y + size // 2], 
                           [x + size // 2, y + size // 2]], np.int32)
            cv2.fillPoly(frame, [pts], color)
    
    def read(self):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        bg_color = (200, 200, 200)
        frame[:] = bg_color
        
        for obj in self.objects:
            obj['x'] += obj['dx']
            obj['y'] += obj['dy']
            
            if obj['x'] < obj['size'] // 2 or obj['x'] > self.width - obj['size'] // 2:
                obj['dx'] *= -1
            if obj['y'] < obj['size'] // 2 or obj['y'] > self.height - obj['size'] // 2:
                obj['dy'] *= -1
            
            if obj['type'] == 'face':
                self._draw_face(frame, int(obj['x']), int(obj['y']), obj['size'])
            elif obj['type'] == 'shape':
                self._draw_shape(frame, obj['shape'], obj['color'], 
                               int(obj['x']), int(obj['y']), obj['size'])
        
        if self.frame_count % 100 == 0:
            self._init_objects()
        
        self.frame_count += 1
        return True, frame
    
    def isOpened(self):
        return True
    
    def release(self):
        pass

if __name__ == '__main__':
    stream = MockVideoStream()
    
    while True:
        ret, frame = stream.read()
        if not ret:
            break
        
        cv2.imshow('Mock Video Stream', frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()