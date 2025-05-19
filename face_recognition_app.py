import face_recognition
import cv2
import numpy as np
import json
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

def put_text_with_unicode(img, text, org, font_size, color):
    # Створюємо зображення PIL з масиву numpy
    img_pil = Image.fromarray(img)
    
    # Створюємо об'єкт для малювання
    draw = ImageDraw.Draw(img_pil)
    
    # Завантажуємо системний шрифт, який підтримує кирилицю
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Малюємо текст
    draw.text(org, text, font=font, fill=color)
    
    # Конвертуємо назад у масив numpy
    return np.array(img_pil)

def draw_face_contour(img, face_landmarks, color, thickness=2):
    # Малюємо контур обличчя
    jaw = face_landmarks['chin']
    for i in range(len(jaw)-1):
        pt1 = (int(jaw[i][0]), int(jaw[i][1]))
        pt2 = (int(jaw[i+1][0]), int(jaw[i+1][1]))
        cv2.line(img, pt1, pt2, color, thickness)
    
    # Малюємо брови
    left_eyebrow = face_landmarks['left_eyebrow']
    right_eyebrow = face_landmarks['right_eyebrow']
    for points in [left_eyebrow, right_eyebrow]:
        for i in range(len(points)-1):
            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[i+1][0]), int(points[i+1][1]))
            cv2.line(img, pt1, pt2, color, thickness)
    
    # Малюємо ніс
    nose_bridge = face_landmarks['nose_bridge']
    nose_tip = face_landmarks['nose_tip']
    for points in [nose_bridge, nose_tip]:
        for i in range(len(points)-1):
            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[i+1][0]), int(points[i+1][1]))
            cv2.line(img, pt1, pt2, color, thickness)
    
    # Малюємо очі
    left_eye = face_landmarks['left_eye']
    right_eye = face_landmarks['right_eye']
    for points in [left_eye, right_eye]:
        for i in range(len(points)):
            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[(i+1)%len(points)][0]), int(points[(i+1)%len(points)][1]))
            cv2.line(img, pt1, pt2, color, thickness)
    
    # Малюємо губи
    top_lip = face_landmarks['top_lip']
    bottom_lip = face_landmarks['bottom_lip']
    for points in [top_lip, bottom_lip]:
        for i in range(len(points)):
            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[(i+1)%len(points)][0]), int(points[(i+1)%len(points)][1]))
            cv2.line(img, pt1, pt2, color, thickness)
    
    return img

def draw_info_panel(img, name, friend_info, x, y, width=300, padding=10):
    # Фон панелі (напівпрозорий)
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + 120), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Додаємо градієнт зліва
    for i in range(5):
        cv2.line(img, (x + i, y), (x + i, y + 120), (0, 150, 255), 1)
    
    # Конвертуємо в RGB для PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Додаємо текст
    if name != "Невідомий":
        # Ім'я (великим шрифтом)
        img_rgb = put_text_with_unicode(img_rgb, name, (x + padding + 5, y + padding), 24, (255, 255, 255))
        
        if friend_info:
            # Вік
            age_text = f"Вік: {friend_info['age']}"
            img_rgb = put_text_with_unicode(img_rgb, age_text, (x + padding + 5, y + padding + 35), 20, (200, 200, 200))
            
            # Інтереси
            interests = friend_info['interests']
            if interests:
                interests_text = f"Інтереси: {', '.join(interests)}"
                img_rgb = put_text_with_unicode(img_rgb, interests_text, (x + padding + 5, y + padding + 65), 18, (200, 200, 200))
    else:
        img_rgb = put_text_with_unicode(img_rgb, "Невідомий", (x + padding + 5, y + padding + 30), 24, (200, 200, 200))
    
    # Конвертуємо назад в BGR
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def load_known_faces():
    with open('info.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    known_face_encodings = []
    known_face_names = []
    
    for name, friend_data in data['friends'].items():
        person_encodings = []
        for photo in friend_data['photos'][:3]:  # Беремо перші 3 фото для кожної людини
            image_path = os.path.join('all face', photo)
            if os.path.exists(image_path):
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_locations = face_recognition.face_locations(image, model="hog")
                    if face_locations:
                        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(name)
                        print(f"Завантажено обличчя {name} з {photo}")
                except Exception as e:
                    print(f"Помилка при обробці зображення {photo}: {str(e)}")
            else:
                print(f"Попередження: Файл {image_path} не знайдено")
    
    return known_face_encodings, known_face_names

def get_friend_info(name):
    with open('info.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['friends'].get(name, {})

def main():
    # Завантаження відомих облич
    known_face_encodings, known_face_names = load_known_faces()
    
    if not known_face_encodings:
        print("Помилка: Не знайдено жодного відомого обличчя")
        return
    
    print(f"Завантажено {len(known_face_names)} відомих облич")
    
    # Ініціалізація веб-камери
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Помилка: Не вдалося відкрити камеру")
        return
    
    # Ініціалізація змінних
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    
    while True:
        # Отримання одного кадру відео
        ret, frame = video_capture.read()
        
        if not ret:
            print("Помилка: Не вдалося отримати кадр з камери")
            break
        
        # Створюємо копію кадру для малювання
        display_frame = frame.copy()
        
        # Зменшення розміру кадру для прискорення обробки
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Конвертація з BGR в RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Обробка кожного другого кадру відео
        if process_this_frame:
            # Знаходження всіх облич та їх ключових точок
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
                
                if True in matches:
                    matched_indexes = [i for i, match in enumerate(matches) if match]
                    distances = face_recognition.face_distance([known_face_encodings[i] for i in matched_indexes], face_encoding)
                    best_match_index = matched_indexes[np.argmin(distances)]
                    confidence = (1 - min(distances)) * 100
                    
                    if confidence > 45:
                        name = known_face_names[best_match_index]
                        print(f"Розпізнано: {name} (впевненість: {confidence:.1f}%)")
                    else:
                        name = "Невідомий"
                        print(f"Низька впевненість: {confidence:.1f}%")
                else:
                    name = "Невідомий"
                    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    confidence = (1 - min(distances)) * 100
                    print(f"Не розпізнано (найближча схожість: {confidence:.1f}%)")
                
                face_names.append(name)
        
        process_this_frame = not process_this_frame
        
        # Відображення результатів
        for (top, right, bottom, left), name, landmarks in zip(face_locations, face_names, face_landmarks_list):
            # Масштабування координат та ключових точок
            scaled_landmarks = {}
            for feature, points in landmarks.items():
                scaled_points = [(int(x * 4), int(y * 4)) for (x, y) in points]
                scaled_landmarks[feature] = scaled_points
            
            # Малюємо контур обличчя
            display_frame = draw_face_contour(display_frame, scaled_landmarks, (0, 150, 255), 2)
            
            # Додаємо інформаційну панель
            panel_x = min(right * 4 + 10, display_frame.shape[1] - 310)
            panel_y = max(top * 4 - 60, 10)
            
            friend_info = get_friend_info(name) if name != "Невідомий" else None
            display_frame = draw_info_panel(display_frame, name, friend_info, panel_x, panel_y)
        
        # Відображення результату
        cv2.imshow('Video', display_frame)
        
        # Натисніть 'q' для виходу
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Звільнення ресурсів
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 