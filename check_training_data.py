import os
import face_recognition
import cv2
import numpy as np

def check_training_data():
    training_dir = "all face"
    if not os.path.exists(training_dir):
        print(f"Помилка: Директорія {training_dir} не знайдена")
        return
    
    print("Перевірка навчальних даних...")
    total_images = 0
    successful_detections = 0
    
    for filename in os.listdir(training_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            total_images += 1
            image_path = os.path.join(training_dir, filename)
            print(f"\nОбробка: {filename}")
            
            # Завантаження зображення
            image = face_recognition.load_image_file(image_path)
            
            # Пошук облич
            face_locations = face_recognition.face_locations(image)
            
            if face_locations:
                successful_detections += 1
                print(f"Знайдено облич: {len(face_locations)}")
            else:
                print("Не знайдено облич")
    
    print(f"\nПідсумок:")
    print(f"Всього зображень: {total_images}")
    print(f"Успішних виявлень: {successful_detections}")
    print(f"Відсоток успішності: {(successful_detections/total_images)*100:.2f}%")

if __name__ == "__main__":
    check_training_data() 