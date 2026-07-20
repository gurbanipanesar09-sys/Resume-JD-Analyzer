import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from tensorflow.keras.models import load_model

from preprocessing import (
    load_dataset,
    split_dataset,
    prepare_inputs,
)

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "resume_jd_model.keras",
)

TOKENIZER_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "tokenizer.pkl",
)

DATASET_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "resume_jd_dataset.csv",
)

OUTPUT_PATH = os.path.join(
    PROJECT_DIR,
    "outputs",
    "graphs",
)

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Loading dataset...")

dataset = load_dataset(DATASET_PATH)

_, _, test_data = split_dataset(dataset)

tokenizer = joblib.load(TOKENIZER_PATH)

test_resume, test_jd, test_labels = prepare_inputs(
    test_data,
    tokenizer,
)

print("Loading trained model...")

model = load_model(
    MODEL_PATH,
    safe_mode=False,
)

loss, accuracy = model.evaluate(
    [test_resume, test_jd],
    test_labels,
    verbose=0,
)

print("\nTest Accuracy:", accuracy)
print("Test Loss:", loss)

predictions = model.predict(
    [test_resume, test_jd],
    verbose=0,
)

predicted_labels = np.argmax(predictions, axis=1)

print("\nClassification Report:\n")

print(
    classification_report(
        test_labels,
        predicted_labels,
        target_names=[
            "Weak Match",
            "Medium Match",
            "Strong Match",
        ],
    )
)

cm = confusion_matrix(
    test_labels,
    predicted_labels,
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Weak",
        "Medium",
        "Strong",
    ],
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "confusion_matrix.png",
    )
)

plt.close()

print("\nConfusion Matrix Saved Successfully.")