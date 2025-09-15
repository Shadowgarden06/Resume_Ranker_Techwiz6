AI Resume Ranker

📂 Category: AI and Machine Learning Mania
👥 Team: DATA UNITED

TRAN CONG KHOI-1604958 (Leader)

DO NGUYET HO TRUC-1611860

NGUYEN DUC TRUONG GIANG-1612595

VO MINH HOANG AN-1595870

🎓 Instructor: HO NHUT MINH
📍 2025, Vietnam

📖 Overview

The AI Resume Ranker is an intelligent system that automates resume screening and candidate ranking using Natural Language Processing (NLP) and Machine Learning (ML).

It parses resumes, extracts skills, experience, and qualifications, then matches them against job descriptions using a hybrid scoring mechanism (TF-IDF, semantic similarity, and skill-based matching).

⚡ Why it matters?

Reduces manual screening time.

Improves candidate-job matching accuracy.

Promotes fairness and reduces bias in recruitment.

📷 Demo & Resources

📹 YouTube Demo

💻 GitHub Repository

📝 Project Blog

🛠️ Installation
Prerequisites

Python 3.9+

Libraries: NLTK, spaCy, scikit-learn, pandas, numpy, flask, streamlit

Steps
# Clone repository
git clone https://github.com/Shadowgarden06/Resume_Ranker_Techwiz6.git
cd Resume_Ranker_Techwiz6

# Create virtual environment
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

🚀 Usage

Upload a resume dataset and a job description.

System parses and extracts entities (skills, roles, experience).

Candidates are ranked automatically based on similarity.

Results are displayed with charts, scores, and export options.

Example:

python main.py --resume data/resumes/ --job data/job_description.txt

📊 Results (Highlights)

Evaluated on 225 resumes.

Best match score: 0.597

Average similarity: 0.414

NER Performance: Precision 0.93 | Recall 0.98 | F1 0.94

Ranking: NDCG = 1.00 (consistent with ground truth).

🤝 Contributing

We welcome contributions!

Fork this repository.

Create a feature branch (git checkout -b feature-name).

Commit changes (git commit -m "Add new feature").

Push branch (git push origin feature-name).

Open a Pull Request.

📜 License

This project is licensed under the MIT License – see the LICENSE
 file for details.

📬 Contact

For questions or collaboration:

Leader: Tran Cong Khoi

Email: congkhoitran01@gmail.com

GitHub: Shadowgarden06
