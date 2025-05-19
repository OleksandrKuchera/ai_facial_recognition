# Face Recognition System

This project is a face recognition system that uses the dlib and face_recognition libraries for Python. The system allows for real-time face recognition and displays information about recognized people.

## Features

- Real-time face recognition
- Storage and display of information about recognized people
- Model training on custom data
- Training data quality verification

## Requirements

- Python 3.7+
- dlib
- face_recognition
- OpenCV
- numpy
- scikit-learn

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download required models:
```bash
python download_models.py
```

## Usage

1. Data Preparation:
   - Create an `info.json` file with information about people
   - File format:
   ```json
   {
       "people": {
           "person_id": {
               "name": "Name",
               "age": Age,
               "interests": ["Interest1", "Interest2"],
               "contact": "Contact Information"
           }
       }
   }
   ```

2. Train the model:
```bash
python train_model.py
```

3. Launch the recognition system:
```bash
python face_recognition_app.py
```

## Project Structure

- `face_recognition_app.py` - main application for face recognition
- `train_model.py` - script for model training
- `check_training_data.py` - utility for training data verification
- `download_models.py` - script for downloading required models
- `info.json` - file containing information about people
- `requirements.txt` - Python dependencies list
