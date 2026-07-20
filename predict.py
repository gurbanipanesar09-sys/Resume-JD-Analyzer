import os

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from preprocessing import (
    clean_text,
    convert_text_to_sequences,
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

OUTPUT_PATH = os.path.join(
    PROJECT_DIR,
    "outputs",
    "predictions",
    "latest_prediction.txt",
)


CLASS_NAMES = {
    0: "Weak Match",
    1: "Medium Match",
    2: "Strong Match",
}


SKILL_LIST = [
    "python",
    "java",
    "c++",
    "sql",
    "excel",
    "power bi",
    "tableau",
    "pandas",
    "numpy",
    "tensorflow",
    "pytorch",
    "machine learning",
    "deep learning",
    "nlp",
    "computer vision",
    "html",
    "css",
    "javascript",
    "react",
    "node.js",
    "express",
    "mongodb",
    "bootstrap",
    "git",
    "rest api",
    "oop",
    "aws",
    "azure",
    "docker",
    "kubernetes",
    "linux",
    "terraform",
    "devops",
    "fastapi",
    "transformers",
    "llm",
    "statistics",
    "data visualization",
    "network security",
    "siem",
    "firewall",
    "ethical hacking",
    "risk assessment",
    "cryptography",
    "penetration testing",
]


def extract_skills(text):
    cleaned_text = clean_text(text)

    detected_skills = {
        skill
        for skill in SKILL_LIST
        if skill in cleaned_text
    }

    return detected_skills


def create_recommendation(predicted_class):
    if predicted_class == 2:
        return (
            "The candidate has strong alignment with the job "
            "requirements and is recommended for the next "
            "interview stage."
        )

    if predicted_class == 1:
        return (
            "The candidate has partial alignment with the job "
            "requirements. A technical screening is recommended."
        )

    return (
        "The candidate has limited alignment with the job "
        "requirements. More relevant skills and experience "
        "may be required."
    )


def predict_match(resume_text, job_description):
    if not resume_text.strip():
        raise ValueError("Resume text cannot be empty.")

    if not job_description.strip():
        raise ValueError(
            "Job description cannot be empty."
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run train.py first."
        )

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            "Tokenizer not found. Run preprocessing.py first."
        )

    model = load_model(
        MODEL_PATH,
        safe_mode=False,
    )

    tokenizer = joblib.load(
        TOKENIZER_PATH
    )

    cleaned_resume = clean_text(
        resume_text
    )

    cleaned_job_description = clean_text(
        job_description
    )

    resume_sequence = convert_text_to_sequences(
        tokenizer,
        [cleaned_resume],
    )

    job_description_sequence = (
        convert_text_to_sequences(
            tokenizer,
            [cleaned_job_description],
        )
    )

    probabilities = model.predict(
        [
            resume_sequence,
            job_description_sequence,
        ],
        verbose=0,
    )[0]

    predicted_class = int(
        np.argmax(probabilities)
    )

    confidence_score = float(
        probabilities[predicted_class]
    )

    resume_skills = extract_skills(
        resume_text
    )

    job_skills = extract_skills(
        job_description
    )

    matched_skills = sorted(
        resume_skills.intersection(job_skills)
    )

    missing_skills = sorted(
        job_skills.difference(resume_skills)
    )

    result = (
        "RESUME-JD MATCH ANALYSIS\n"
        + "=" * 45
        + f"\nPredicted Match: "
        f"{CLASS_NAMES[predicted_class]}"
        + f"\nConfidence Score: "
        f"{confidence_score * 100:.2f}%"
        + "\n\nClass Probabilities:"
        + f"\nWeak Match: "
        f"{probabilities[0] * 100:.2f}%"
        + f"\nMedium Match: "
        f"{probabilities[1] * 100:.2f}%"
        + f"\nStrong Match: "
        f"{probabilities[2] * 100:.2f}%"
        + "\n\nMatched Skills:"
        + (
            "\n- " + "\n- ".join(matched_skills)
            if matched_skills
            else "\nNo matched skills detected."
        )
        + "\n\nMissing Skills:"
        + (
            "\n- " + "\n- ".join(missing_skills)
            if missing_skills
            else "\nNo missing skills detected."
        )
        + "\n\nRecommendation:\n"
        + create_recommendation(
            predicted_class
        )
    )

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(result)

    return result


def main():
    print(
        "\nINTELLIGENT RESUME-JD MATCHING SYSTEM"
    )

    print("=" * 45)

    print(
        "\nPaste the resume text and press Enter."
    )

    resume_text = input(
        "\nResume: "
    )

    print(
        "\nPaste the job description and press Enter."
    )

    job_description = input(
        "\nJob Description: "
    )

    result = predict_match(
        resume_text,
        job_description,
    )

    print("\n" + result)

    print(
        "\nPrediction saved at:"
    )

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()