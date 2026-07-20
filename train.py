import os
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)

from preprocessing import (
    load_dataset,
    split_dataset,
    create_tokenizer,
    prepare_inputs,
)

from model import build_siamese_bilstm


PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "resume_jd_dataset.csv",
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models",
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "graphs",
)

TOKENIZER_PATH = os.path.join(
    MODEL_DIR,
    "tokenizer.pkl",
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "resume_jd_model.keras",
)


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


print("Loading dataset...")

dataset = load_dataset(DATASET_PATH)

train_data, validation_data, test_data = split_dataset(dataset)

print("Creating tokenizer...")

tokenizer = create_tokenizer(train_data)

joblib.dump(
    tokenizer,
    TOKENIZER_PATH,
)

print("Preparing sequences...")

train_resume, train_jd, train_labels = prepare_inputs(
    train_data,
    tokenizer,
)

validation_resume, validation_jd, validation_labels = prepare_inputs(
    validation_data,
    tokenizer,
)

print("Building model...")

model = build_siamese_bilstm()

model.summary()

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    ),

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
    )

]

print("\nTraining started...\n")

history = model.fit(

    [train_resume, train_jd],

    train_labels,

    validation_data=(

        [validation_resume, validation_jd],

        validation_labels,

    ),

    epochs=10,

    batch_size=32,

    callbacks=callbacks,

)

print("\nTraining Completed!")

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
)

plt.legend()

plt.title("Training Accuracy")

plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "accuracy.png",

    )

)

plt.close()


plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Training Loss",
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
)

plt.legend()

plt.title("Training Loss")

plt.savefig(

    os.path.join(

        OUTPUT_DIR,

        "loss.png",

    )

)

plt.close()

print("\nGraphs Saved Successfully.")

print("Model Saved Successfully.")