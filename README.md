# ProctorSense: AI-Based Online Exam Proctoring

ProctorSense is a machine learning project that flags suspicious behaviour during online exams. It looks at face, head pose, gaze, hand and phone-related features, trains a Random Forest to classify each sample as **normal** or **suspicious**, and combines that result with audio, object and face checks into one proctoring decision.

## Why this project

Online exams are hard to supervise. Many proctoring tools produce false alarms, miss subtle cheating, and watch only one signal. ProctorSense uses several signals together:

- Face detection
- Head pose and gaze direction
- Phone presence
- Hand movement and hand-object interaction

## What is in the notebook

| Part | What it does |
|------|--------------|
| Behaviour model | Loads `Students.csv` (5,500 samples, 38 features), explores the data, trains a Random Forest (100 trees, 75/25 train-test split) and plots a confusion matrix and the top 10 important features. |
| Audio module | Loads the UrbanSound8K metadata (8,732 clips), cleans it, encodes classes, scales numeric features and plots class distribution and a correlation heatmap. |
| Object module | Loads the object-detection labels (book, cell phone, headphone, laptop, person, TV), cleans and encodes them. |
| Face module | Loads face-detection image and label records, encodes labels and plots the label distribution. |
| Decision logic | A rule-based function combines the four module flags into one result, such as "Suspicious Behavior Detected" or "Normal Exam Behavior". |

## How the final decision works

```text
behaviour prediction ─┐
audio flag ───────────┤
object flag ──────────┼──► proctoring_decision() ──► Normal / Suspicious (with reason)
face flag ────────────┘
```

The checks run in priority order: behaviour, then audio, then object, then face. The first module that raises a flag decides the message. If none do, the exam behaviour is reported as normal.

## Tech stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Google Colab / Jupyter. YOLOv8n weights are used for the object-detection side.

## Datasets

The data files are not stored in this repo because of their size. Add them to the project folder before running:

| File | Used for |
|------|----------|
| `Students.csv` | Behaviour classification (target column: `label`) |
| `UrbanSound8K.csv` | Audio metadata |
| `object.csv.xlsx` | Object presence labels |
| `face_detection.csv` | Face image and label records |

## Getting started

```bash
git clone https://github.com/<your-username>/proctorsense.git
cd proctorsense
pip install -r requirements.txt
jupyter notebook MLproject.ipynb
```

The notebook was written in Google Colab. If you run it locally, replace the `files.upload()` cells with `pd.read_csv("<path-to-file>")`.

## Results

The Random Forest is evaluated with a confusion matrix and a feature-importance chart (see the notebook outputs).

> Add your test accuracy here after running the notebook, for example: `Test accuracy: XX.X%`

## Limitations and next steps

- The audio, object and face modules currently cover data cleaning and exploration. Their flags in the final decision step are simulated examples, not live model predictions.
- Train and plug in a classifier for each module (for example a YOLO detector for phones and a sound-event classifier for audio).
- Test on live webcam and microphone input.
- Compare Random Forest with other models such as Gradient Boosting or XGBoost.

## Author

**Mitali Sabale**
[LinkedIn](https://www.linkedin.com/in/mitalisabale)

If you built this with teammates, list them here.
