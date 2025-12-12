# Social Media Sentiment Analysis 🎭

A comprehensive sentiment analysis project using Natural Language Processing to classify social media text sentiment (positive, negative, neutral).

## 🌟 Features

- **Multiple ML Models**: Logistic Regression & Random Forest
- **Advanced NLP**: Text preprocessing, tokenization, lemmatization
- **TF-IDF Features**: Advanced feature engineering
- **Visualizations**: Confusion matrices, performance comparisons
- **Production Ready**: Clean, modular code structure

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/kkaufma72/Social-Media-Sentiment-Analysis.git
cd Social-Media-Sentiment-Analysis

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

### Run Demo
```bash
python demo.py
```

## 📁 Project Structure
```
Social-Media-Sentiment-Analysis/
├── src/
│   ├── data_preprocessing.py    # Text cleaning & preprocessing
│   ├── feature_engineering.py   # TF-IDF feature extraction
│   ├── train_models.py          # Model training
│   └── evaluate.py              # Evaluation & visualization
├── data/                        # Dataset storage
├── models/                      # Saved models
├── notebooks/                   # Jupyter notebooks
├── results/                     # Output visualizations
├── demo.py                      # Quick demo script
└── requirements.txt             # Dependencies
```

## 📊 Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~82% |
| Random Forest | ~79% |

## 🛠️ Technologies

- Python 3.8+
- scikit-learn
- NLTK
- Pandas & NumPy
- Matplotlib & Seaborn

## 📝 Usage Example
```python
from src.data_preprocessing import TextPreprocessor
from src.train_models import SentimentModelTrainer

# Preprocess text
preprocessor = TextPreprocessor()
clean_text = preprocessor.preprocess("I love this product!")

# Train model
trainer = SentimentModelTrainer('logistic_regression')
trainer.train(X_train, y_train)

# Make predictions
prediction = trainer.predict(X_test)
```

## 🤝 Contributing

Contributions welcome! Feel free to open issues or submit PRs.

## 📧 Contact

Kyle Kaufman - [GitHub](https://github.com/kkaufma72)

## 📄 License

MIT License - see LICENSE file for details
