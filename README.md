# HandSigns_to_Text_conversion

This project enables real-time hand sign detection and conversion into text using a machine learning model trained on hand landmarks captured via MediaPipe.
---
**Overview**
- Detects hand gestures (A–Z, SPACE, NEXTLINE, DELETE)
- Normalizes hand positions for consistent predictions
- Requires holding a gesture steadily to confirm input
- Displays real-time predictions and builds a live text stream

**Project Structure**
```
hand-sign-to-text/
│
├── Handsign_detection.py      # Real-time hand sign detection
├── model_pkl.txt              # Trained model file (pickle)
├── requirements.txt           # Required Python packages
└── Handsigns_data.csv         # Optional dataset for training/debugging
```
---
## Getting Started
---
1. **Clone the Repository**
```
git clone https://github.com/yourusername/hand-sign-to-text.git
cd hand-sign-to-text
```

2. **Install Dependencies**
```
pip install -r requirements.txt
```

3. **Run the Application**
```
python handsign_detection.py
```

> Press `q` to quit the live video stream.

### How It Works
- Uses MediaPipe Hands to detect 21 hand landmarks
- Normalizes landmark vectors relative to the wrist to remove positional bias
- Sends the landmark vector to the trained model.
- Uses a timing-based gesture confirmation system to avoid accidental triggers
- Displays predictions in real time and constructs a final output text

**Model Training (optional)**
- If you want to train your own model using hand_signs2.csv:
- Each data row must contain 63 landmark features + label
- Valid labels include A–Z, SPACE, NextLine, Delete
- You may train using:
1. RandomForest
2. SVM
- Optuna-optimized ML models

**Example Output**

A sample demonstration is available here:
[https://www.linkedin.com/posts/kalyani-gajjala_python-machinelearning-opencv-ugcPost-7358815324005126147-KokA?utm_source=share&utm_medium=member_desktop&rcm=ACoAADe5fZgB8oJyT9gL9x2OWXwW8HOjWv9auEk]

**Dependencies**
```
opencv-python
mediapipe
scikit-learn
numpy
```
