import dlib
import cv2
import numpy as np
from PIL import Image
import json
import os
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def load_friends_info():
    with open('info.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def load_image_file(file):
    """Завантажує зображення за допомогою PIL і конвертує його в масив numpy"""
    im = Image.open(file)
    im = im.convert('RGB')
    return np.array(im)

def preprocess_image(image):
    # Конвертація в сіре зображення
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Покращення контрасту
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Конвертація назад в RGB
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced_rgb

def get_face_encoding(face_detector, shape_predictor, face_recognition_model, image):
    """Отримує кодування обличчя за допомогою dlib"""
    # Знаходження обличь на зображенні
    dets = face_detector(image, 1)
    
    if len(dets) == 0:
        return None
    
    # Отримання ключових точок обличчя
    shape = shape_predictor(image, dets[0])
    
    # Отримання кодування обличчя
    face_descriptor = np.array(face_recognition_model.compute_face_descriptor(image, shape))
    
    return face_descriptor

def train_face_recognition_model(epochs=10):
    print("Початок навчання моделі...")
    
    # Ініціалізація детекторів та моделей dlib
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    
    # Завантаження інформації про друзів
    friends_info = load_friends_info()
    
    # Списки для зберігання даних
    face_encodings = []
    face_names = []
    
    # Збір даних з фотографій
    for filename in friends_info['friends'].keys():
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            print(f"Обробка фотографії: {filename}")
            
            # Завантаження зображення
            image_path = os.path.join('all face', filename)
            if not os.path.exists(image_path):
                print(f"Увага: Файл {filename} не знайдено")
                continue
                
            image = load_image_file(image_path)
            
            # Попередня обробка зображення
            processed_image = preprocess_image(image)
            
            # Отримання кодування обличчя
            face_encoding = get_face_encoding(face_detector, shape_predictor, face_recognition_model, processed_image)
            
            if face_encoding is None:
                print(f"Увага: Не знайдено обличчя на фотографії {filename}")
                continue
            
            # Отримання імені з інформації про друга
            name = friends_info['friends'][filename]['name']
            
            # Додавання даних до списків
            face_encodings.append(face_encoding)
            face_names.append(name)
    
    if len(face_encodings) == 0:
        print("Помилка: Не знайдено жодного обличчя для навчання")
        return
    
    # Конвертація списків у numpy масиви
    X = np.array(face_encodings)
    
    # Кодування міток (імен)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(face_names)
    
    # Перевірка кількості зразків для кожного класу
    unique_classes, counts = np.unique(y, return_counts=True)
    print("\nКількість зразків для кожного класу:")
    for cls, count in zip(label_encoder.classes_, counts):
        print(f"{cls}: {count}")
    
    # Видалення класів з менше ніж 2 зразками
    valid_classes = unique_classes[counts >= 2]
    if len(valid_classes) < 2:
        print("Помилка: Недостатньо класів з мінімум 2 зразками")
        return
    
    # Фільтрація даних
    mask = np.isin(y, valid_classes)
    X = X[mask]
    y = y[mask]
    
    # Оновлення класів у кодувальнику
    label_encoder.fit(np.array(face_names)[mask])
    
    # Розділення на навчальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Параметри для пошуку найкращих гіперпараметрів
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    # Пошук найкращих параметрів
    print("\nПошук найкращих параметрів...")
    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Найкращі параметри: {grid_search.best_params_}")
    
    # Навчання з найкращими параметрами
    best_classifier = grid_search.best_estimator_
    
    # Оцінка на тестовому наборі
    y_pred = best_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nЗвіт про класифікацію:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Збереження навченої моделі та кодувальника міток
    print("\nЗбереження моделі...")
    with open('face_recognition_model.pkl', 'wb') as f:
        pickle.dump((best_classifier, label_encoder), f)
    
    # Збереження метрик
    metrics = {
        'accuracy': float(accuracy),
        'best_params': {str(k): str(v) for k, v in grid_search.best_params_.items()},
        'total_samples': int(len(X_train) + len(X_test)),
        'unique_individuals': int(len(np.unique(y_train))),
        'test_samples': int(len(X_test))
    }
    
    with open('model_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    
    print("\nНавчання завершено!")
    print(f"Загальна кількість навчених зразків: {len(face_encodings)}")
    print(f"Кількість унікальних осіб: {len(np.unique(face_names))}")
    print(f"Точність на тестовому наборі: {accuracy:.4f}")

if __name__ == "__main__":
    train_face_recognition_model(epochs=10) 