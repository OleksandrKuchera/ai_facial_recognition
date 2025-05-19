import os
import urllib.request
import bz2

def download_file(url, filename):
    print(f"Завантаження {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Завантажено {filename}")

def extract_bz2(filename):
    print(f"Розпакування {filename}...")
    with bz2.BZ2File(filename) as fr, open(filename[:-4], 'wb') as fw:
        fw.write(fr.read())
    os.remove(filename)
    print(f"Розпаковано {filename}")

def main():
    # URLs для моделей
    shape_predictor_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    recognition_model_url = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
    
    # Завантаження та розпакування shape predictor
    download_file(shape_predictor_url, "shape_predictor_68_face_landmarks.dat.bz2")
    extract_bz2("shape_predictor_68_face_landmarks.dat.bz2")
    
    # Завантаження та розпакування recognition model
    download_file(recognition_model_url, "dlib_face_recognition_resnet_model_v1.dat.bz2")
    extract_bz2("dlib_face_recognition_resnet_model_v1.dat.bz2")
    
    print("Всі моделі завантажено та розпаковано!")

if __name__ == "__main__":
    main() 