import os
import random

import pandas as pd


random.seed(42)


ROLE_SKILLS = {
    "Data Scientist": [
        "Python",
        "Machine Learning",
        "Deep Learning",
        "TensorFlow",
        "Pandas",
        "NumPy",
        "SQL",
        "Statistics",
        "Data Visualization",
        "Scikit-learn",
    ],
    "Data Analyst": [
        "Python",
        "Excel",
        "SQL",
        "Power BI",
        "Tableau",
        "Statistics",
        "Data Cleaning",
        "Pandas",
        "Business Intelligence",
        "Reporting",
    ],
    "Software Engineer": [
        "Java",
        "Python",
        "C++",
        "Git",
        "Algorithms",
        "Data Structures",
        "REST API",
        "OOP",
        "Debugging",
        "Testing",
    ],
    "Web Developer": [
        "HTML",
        "CSS",
        "JavaScript",
        "React",
        "Node.js",
        "Express",
        "MongoDB",
        "Bootstrap",
        "Git",
        "REST API",
    ],
    "AI Engineer": [
        "Python",
        "TensorFlow",
        "PyTorch",
        "LLM",
        "NLP",
        "Computer Vision",
        "FastAPI",
        "Docker",
        "Deep Learning",
        "Transformers",
    ],
    "Cloud Engineer": [
        "AWS",
        "Azure",
        "Docker",
        "Kubernetes",
        "Linux",
        "Terraform",
        "CI/CD",
        "Networking",
        "DevOps",
        "Python",
    ],
    "Cybersecurity Analyst": [
        "Network Security",
        "Linux",
        "Python",
        "SIEM",
        "Firewall",
        "Incident Response",
        "Ethical Hacking",
        "Risk Assessment",
        "Cryptography",
        "Penetration Testing",
    ],
}


EDUCATION = [
    "Bachelor's degree in Computer Science",
    "Bachelor's degree in Information Technology",
    "Bachelor's degree in Data Science",
    "Bachelor's degree in Engineering",
    "Master's degree in Artificial Intelligence",
    "Bachelor's degree in Computational Statistics and Data Analytics",
]


EXPERIENCE = [
    "Fresher",
    "6 months internship",
    "1 year experience",
    "2 years experience",
    "3 years experience",
]


SOFT_SKILLS = [
    "Communication",
    "Leadership",
    "Problem Solving",
    "Teamwork",
    "Time Management",
    "Analytical Thinking",
    "Critical Thinking",
]


def create_resume(role, skills):
    education = random.choice(EDUCATION)
    experience = random.choice(EXPERIENCE)
    soft_skills = random.sample(SOFT_SKILLS, 2)

    resume = f"""
Candidate applying for {role}.

Education:
{education}

Experience:
{experience}

Technical Skills:
{", ".join(skills)}

Soft Skills:
{", ".join(soft_skills)}

Completed several projects related to {role}.
"""

    return resume.strip()


def create_job_description(role, skills):
    soft_skills = random.sample(SOFT_SKILLS, 2)

    job_description = f"""
We are hiring a {role}.

Required Skills:
{", ".join(skills)}

Preferred Soft Skills:
{", ".join(soft_skills)}

The candidate should work on real-world projects and collaborate
with team members.
"""

    return job_description.strip()


def create_dataset_pair(label):
    roles = list(ROLE_SKILLS.keys())

    if label == 2:
        role = random.choice(roles)
        skills = ROLE_SKILLS[role]

        common_skills = random.sample(skills, 7)

        resume = create_resume(role, common_skills)
        job_description = create_job_description(
            role,
            common_skills,
        )

        return (
            resume,
            job_description,
            role,
            common_skills,
            2,
            "Strong Match",
        )

    if label == 1:
        role = random.choice(roles)
        skills = ROLE_SKILLS[role]

        common_skills = random.sample(skills, 4)

        remaining_skills = [
            skill for skill in skills
            if skill not in common_skills
        ]

        resume_skills = (
            common_skills
            + random.sample(remaining_skills, 1)
        )

        job_skills = (
            common_skills
            + random.sample(remaining_skills, 3)
        )

        resume = create_resume(role, resume_skills)
        job_description = create_job_description(
            role,
            job_skills,
        )

        return (
            resume,
            job_description,
            role,
            common_skills,
            1,
            "Medium Match",
        )

    resume_role, job_role = random.sample(roles, 2)

    resume_skills = random.sample(
        ROLE_SKILLS[resume_role],
        7,
    )

    job_skills = random.sample(
        ROLE_SKILLS[job_role],
        7,
    )

    resume = create_resume(
        resume_role,
        resume_skills,
    )

    job_description = create_job_description(
        job_role,
        job_skills,
    )

    return (
        resume,
        job_description,
        resume_role,
        [],
        0,
        "Weak Match",
    )


def generate_dataset(rows_per_class=1000):
    records = []

    for label in [0, 1, 2]:
        for _ in range(rows_per_class):
            (
                resume,
                job_description,
                role,
                matched_skills,
                numeric_label,
                match_level,
            ) = create_dataset_pair(label)

            records.append(
                {
                    "resume": resume,
                    "job_description": job_description,
                    "role": role,
                    "matched_skills": ", ".join(
                        matched_skills
                    ),
                    "label": numeric_label,
                    "match_level": match_level,
                }
            )

    random.shuffle(records)

    dataset = pd.DataFrame(records)

    return dataset


def main():
    dataset = generate_dataset(rows_per_class=1000)

    current_file_path = os.path.abspath(__file__)
    src_folder = os.path.dirname(current_file_path)
    project_folder = os.path.dirname(src_folder)

    data_folder = os.path.join(
        project_folder,
        "data",
    )

    os.makedirs(
        data_folder,
        exist_ok=True,
    )

    output_path = os.path.join(
        data_folder,
        "resume_jd_dataset.csv",
    )

    dataset.to_csv(
        output_path,
        index=False,
    )

    print("Dataset generated successfully.")
    print("Dataset saved at:")
    print(output_path)

    print("\nTotal rows:")
    print(len(dataset))

    print("\nClass distribution:")
    print(dataset["match_level"].value_counts())

    print("\nDataset columns:")
    print(dataset.columns.tolist())


if __name__ == "__main__":
    main()