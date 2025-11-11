🧠 Multimodal Machine Learning for Product Price Prediction and Classification
📄 Overview

This project implements a multimodal machine learning system that integrates textual and visual features to predict product prices and categorize them into Budget, Mid-Range, and Premium tiers.
It uses data from the Amazon ML Challenge 2025, which includes product descriptions, images, and prices.

The pipeline combines TF-IDF text embeddings and ResNet-50 image embeddings, then applies various machine learning algorithms for regression and classification tasks.
A custom quantile-based sampling algorithm is also implemented to extract a statistically representative subset from the full dataset.

🚀 Key Features

🔤 Text Embedding using TF-IDF – Converts textual product descriptions into feature vectors.

🖼️ Image Embedding using ResNet-50 – Extracts 2048-dimensional image representations using a pre-trained CNN.

⚙️ Feature Fusion – Combines text and image vectors into a single multimodal feature space.

🤖 Model Training – Includes models for both regression and classification:

Regression: Linear, Decision Tree, Random Forest, SVR, KNN, XGBoost, LightGBM

Classification: Logistic Regression, Decision Tree, Random Forest, SVC, KNN, Naïve Bayes, XGBoost, LightGBM

📊 Evaluation Metrics: MAE, RMSE, R², Accuracy, Precision, Recall, F1-score, ROC-AUC

📈 Visualizations: Confusion Matrix, ROC & PR Curves, Predicted vs Actual Scatterplots

📉 Statistical Sampling: Stratified sampling based on price quantiles to generate a smaller 10,000-sample dataset with identical statistical characteristics.

🧩 System Architecture

🧪 Model Pipeline

📊 Results Comparison

Task	Best Model	Metric	Score
Regression	XGBoost Regressor	R²	0.023
Classification	KNN Classifier (k=7)	Accuracy	45%
🧠 How It Works
1️⃣ Data Preprocessing

Load dataset (sample_id, catalog_content, image_link, price)

Clean text (lowercase, remove missing entries)

Convert price to numeric and handle missing values

2️⃣ Feature Engineering

Textual: TF-IDF Vectorizer (max_features=100000, ngram_range=(1,2))

Visual: ResNet-50 embeddings extracted using PyTorch

Fusion: Concatenate both feature types horizontally

3️⃣ Model Training

Trains models separately for:

Regression → predicts continuous price

Classification → predicts price tier (Budget, Mid-Range, Premium)

4️⃣ Evaluation

Uses Scikit-learn metrics for both regression and classification:

Regression: MAE, RMSE, R²

Classification: Accuracy, Precision, Recall, F1, Confusion Matrix

5️⃣ Statistical Sampling

Generates a smaller dataset with similar distribution:

Computes mean, median, variance, std, skewness, kurtosis

Samples 10,000 rows with matching quantile bins

Saves as new_dataset.csv

🧰 Tech Stack
Category	Tools / Libraries
Language	Python 3.9+
Libraries	Scikit-learn, PyTorch, Torchvision, XGBoost, LightGBM, Pandas, NumPy, Matplotlib
Environment	Jupyter Notebook
Visualization	Matplotlib
ML Frameworks	XGBoost, LightGBM, Scikit-learn
🧾 Project Structure
📦 Multimodal_Price_Prediction
 ┣ 📂 Figures/
 ┃ ┣ 📜 fig1_system_architecture.png
 ┃ ┣ 📜 fig2_model_pipeline.png
 ┃ ┗ 📜 fig3_results_comparison.png
 ┣ 📜 IEEE_Conference_Paper.docx
 ┣ 📜 IEEE_Conference_Paper.tex
 ┣ 📜 new_dataset.csv
 ┣ 📜 README.md
 ┣ 📜 amazon_ml_pipeline.ipynb
 ┗ 📜 requirements.txt

⚙️ Installation & Usage
1. Clone Repository
git clone https://github.com/<your_username>/Multimodal-Price-Prediction.git
cd Multimodal-Price-Prediction

2. Install Dependencies
pip install -r requirements.txt

3. Run Jupyter Notebook
jupyter notebook amazon_ml_pipeline.ipynb

4. (Optional) Generate New Dataset
python sample_generation.py

📚 Results Summary
Model	Task	Performance	Interpretation
XGBoost Regressor	Regression	RMSE=23.21, R²=0.023	Best predictor of continuous prices
KNN (k=7)	Classification	Accuracy=45%, F1=0.36	Best at distinguishing price tiers
LightGBM	Regression/Classification	Moderate	Slight overfitting, underperformed on sparse features
Decision Tree	Both	Decent	Good interpretability
Random Forest	Both	Stable	Balanced results, moderate accuracy
🧾 Research Paper

The project report is formatted according to IEEE conference standards and includes:

Abstract, Introduction, Methodology, Results, and Conclusion

Mathematical representation of algorithms

Figures and tables

References in IEEE citation style

📄 Files included:

IEEE_Conference_Paper.docx

IEEE_Conference_Paper.tex

🧠 Future Enhancements

Replace TF-IDF with BERT or Sentence-BERT embeddings for contextual understanding.

Replace ResNet-50 with CLIP or Vision Transformer (ViT) models.

Use Multimodal Transformers (e.g., MMBT, ViLT) for joint training.

Implement automated hyperparameter optimization with Optuna or Ray Tune.

🧑‍💻 Author

Aayush Suthar
B.Tech, Artificial Intelligence and Machine Learning
School of AI & ML, Manipal University Jaipur
📧 aayushsuthar5115@gmail.com
