import os
import re

import joblib
import nltk
import pandas as pd

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# Download English stopwords only if they are not already available
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))


VOCAB_SIZE = 10000
MAX_LENGTH = 150
OOV_TOKEN = "<OOV>"


def clean_text(text):
    """
    Clean resume or job-description text.

    Steps:
    1. Convert text to lowercase.
    2. Remove web links.
    3. Remove unwanted symbols.
    4. Remove extra spaces.
    5. Remove common English stopwords.
    """

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9+#.\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    words = text.split()

    cleaned_words = [
        word
        for word in words
        if word not in STOPWORDS
    ]

    return " ".join(cleaned_words).strip()


def load_dataset(dataset_path):
    """
    Load and validate the dataset.
    """

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_path}"
        )

    dataset = pd.read_csv(dataset_path)

    required_columns = {
        "resume",
        "job_description",
        "label",
    }

    missing_columns = required_columns.difference(
        dataset.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    dataset = dataset.dropna(
        subset=[
            "resume",
            "job_description",
            "label",
        ]
    ).copy()

    dataset["resume"] = dataset[
        "resume"
    ].apply(clean_text)

    dataset["job_description"] = dataset[
        "job_description"
    ].apply(clean_text)

    dataset["label"] = dataset[
        "label"
    ].astype(int)

    return dataset


def split_dataset(dataset):
    """
    Split dataset into:
    70% training
    15% validation
    15% testing
    """

    train_data, temporary_data = train_test_split(
        dataset,
        test_size=0.30,
        random_state=42,
        stratify=dataset["label"],
    )

    validation_data, test_data = train_test_split(
        temporary_data,
        test_size=0.50,
        random_state=42,
        stratify=temporary_data["label"],
    )

    return (
        train_data,
        validation_data,
        test_data,
    )


def create_tokenizer(train_data):
    """
    Create one shared tokenizer for resumes and job descriptions.
    """

    tokenizer = Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token=OOV_TOKEN,
    )

    combined_text = (
        train_data["resume"].tolist()
        + train_data["job_description"].tolist()
    )

    tokenizer.fit_on_texts(combined_text)

    return tokenizer


def convert_text_to_sequences(
    tokenizer,
    text_values,
):
    """
    Convert text into padded integer sequences.
    """

    sequences = tokenizer.texts_to_sequences(
        text_values
    )

    padded_sequences = pad_sequences(
        sequences,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post",
    )

    return padded_sequences


def prepare_inputs(
    dataset,
    tokenizer,
):
    """
    Convert resume and job-description columns into model inputs.
    """

    resume_sequences = convert_text_to_sequences(
        tokenizer,
        dataset["resume"],
    )

    job_description_sequences = (
        convert_text_to_sequences(
            tokenizer,
            dataset["job_description"],
        )
    )

    labels = dataset["label"].values

    return (
        resume_sequences,
        job_description_sequences,
        labels,
    )


def save_tokenizer(
    tokenizer,
    tokenizer_path,
):
    """
    Save the fitted tokenizer.
    """

    tokenizer_folder = os.path.dirname(
        tokenizer_path
    )

    os.makedirs(
        tokenizer_folder,
        exist_ok=True,
    )

    joblib.dump(
        tokenizer,
        tokenizer_path,
    )


def load_tokenizer(tokenizer_path):
    """
    Load a previously saved tokenizer.
    """

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at: {tokenizer_path}"
        )

    tokenizer = joblib.load(tokenizer_path)

    return tokenizer


def main():
    """
    Test the preprocessing pipeline.
    """

    current_file_path = os.path.abspath(
        __file__
    )

    src_folder = os.path.dirname(
        current_file_path
    )

    project_folder = os.path.dirname(
        src_folder
    )

    dataset_path = os.path.join(
        project_folder,
        "data",
        "resume_jd_dataset.csv",
    )

    tokenizer_path = os.path.join(
        project_folder,
        "models",
        "tokenizer.pkl",
    )

    print("Loading dataset...")

    dataset = load_dataset(
        dataset_path
    )

    print("Dataset loaded successfully.")
    print("Total rows:", len(dataset))

    (
        train_data,
        validation_data,
        test_data,
    ) = split_dataset(dataset)

    print("\nDataset split completed.")

    print(
        "Training rows:",
        len(train_data),
    )

    print(
        "Validation rows:",
        len(validation_data),
    )

    print(
        "Testing rows:",
        len(test_data),
    )

    tokenizer = create_tokenizer(
        train_data
    )

    save_tokenizer(
        tokenizer,
        tokenizer_path,
    )

    (
        train_resume,
        train_job_description,
        train_labels,
    ) = prepare_inputs(
        train_data,
        tokenizer,
    )

    (
        validation_resume,
        validation_job_description,
        validation_labels,
    ) = prepare_inputs(
        validation_data,
        tokenizer,
    )

    (
        test_resume,
        test_job_description,
        test_labels,
    ) = prepare_inputs(
        test_data,
        tokenizer,
    )

    print("\nTraining input shapes:")

    print(
        "Resume:",
        train_resume.shape,
    )

    print(
        "Job description:",
        train_job_description.shape,
    )

    print(
        "Labels:",
        train_labels.shape,
    )

    print("\nValidation input shapes:")

    print(
        "Resume:",
        validation_resume.shape,
    )

    print(
        "Job description:",
        validation_job_description.shape,
    )

    print(
        "Labels:",
        validation_labels.shape,
    )

    print("\nTesting input shapes:")

    print(
        "Resume:",
        test_resume.shape,
    )

    print(
        "Job description:",
        test_job_description.shape,
    )

    print(
        "Labels:",
        test_labels.shape,
    )

    print("\nTokenizer vocabulary size:")

    actual_vocabulary_size = len(
        tokenizer.word_index
    ) + 1

    print(actual_vocabulary_size)

    print("\nTokenizer saved at:")
    print(tokenizer_path)

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()