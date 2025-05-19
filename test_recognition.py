import face_recognition
import cv2
import numpy as np
import pickle
import os

def test_recognition():
    # Завантаження навченої моделі
    with open('face_recognition_model.pkl', 'rb') as f:
        classifier, label_encoder = pickle.load(f)
    
    # Завантаження тестового зображення
    test_image = face_recognition.load_image_file("all face/SERGEY.png")
    face_locations = face_recognition.face_locations(test_image)
    
    if not face_locations:
        print("Не знайдено обличчя на зображенні")
        return
    
    # Отримання кодування обличчя
    face_encoding = face_recognition.face_encodings(test_image, face_locations)[0]
    
    # Прогнозування
    predictions = classifier.predict_proba([face_encoding])[0]
    best_class_idx = np.argmax(predictions)
    confidence = predictions[best_class_idx]
    name = label_encoder.inverse_transform([best_class_idx])[0]
    
    print(f"Розпізнано: {name}")
    print(f"Впевненість: {confidence:.2f}")
    
    # Відображення результату
    top, right, bottom, left = face_locations[0]
    cv2.rectangle(test_image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(test_image, f"{name} ({confidence:.2f})", (left, top - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Збереження результату
    cv2.imwrite("test_result.jpg", cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
    print("Результат збережено у файл test_result.jpg")

if __name__ == "__main__":
    test_recognition() 