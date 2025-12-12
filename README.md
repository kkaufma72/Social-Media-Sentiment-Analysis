# Social Media Sentiment Analysis 🎭

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive sentiment analysis project using Natural Language Processing to classify social media text sentiment (positive, negative, neutral).

## 🌟 Features

- **Multiple ML Models**: Logistic Regression & Random Forest
- **Advanced NLP**: Text preprocessing, tokenization, lemmatization
- **TF-IDF Features**: Advanced feature engineering with n-grams
- **Robust Error Handling**: Comprehensive input validation
- **Model Persistence**: Save/load models with metadata
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

# Download NLTK data (handled automatically on first run)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

### Run Demo
```bash
# Basic demo
python demo.py

# With options
python demo.py --size large --model lr --test
```

**Demo Arguments:**
- `--size`: Dataset size (small/medium/large)
- `--model`: Model choice (lr/rf/all)
- `--test`: Run predictions on sample texts

## 📁 Project Structure
```
Social-Media-Sentiment-Analysis/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── data_preprocessing.py    # Text cleaning & preprocessing
│   ├── feature_engineering.py   # TF-IDF feature extraction
│   ├── train_models.py          # Model training
│   ├── evaluate.py              # Evaluation & visualization
│   └── model_utils.py           # Model persistence utilities
├── data/                        # Dataset storage
├── models/                      # Saved models
├── notebooks/                   # Jupyter notebooks
├── results/                     # Output visualizations
├── demo.py                      # Quick demo script
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

## 📊 Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | ~82% | Fast, interpretable |
| Random Forest | ~79% | Robust, ensemble method |

## 🛠️ Technologies

- **Python 3.8+**
- **scikit-learn** - Machine learning models
- **NLTK** - Natural language processing
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Visualization

## 📝 Usage Example
```python
from src.data_preprocessing import TextPreprocessor
from src.train_models import SentimentModelTrainer
from src.feature_engineering import FeatureExtractor

# Preprocess text
preprocessor = TextPreprocessor()
clean_text = preprocessor.preprocess("I love this product!")

# Extract features
extractor = FeatureExtractor()
features = extractor.fit_transform([clean_text])

# Train model
trainer = SentimentModelTrainer('logistic_regression')
trainer.train(X_train, y_train)

# Make predictions
prediction = trainer.predict(features)
```

## 🔬 Advanced Features

### Model Persistence
```python
from src.model_utils import save_model_with_metadata, load_model_with_metadata

# Save with metadata
metadata = {'accuracy': 0.82, 'date': '2025-12-11'}
save_model_with_metadata(model, 'models/sentiment_lr.pkl', metadata)

# Load
model, metadata = load_model_with_metadata('models/sentiment_lr.pkl')
```

### Custom Preprocessing
```python
preprocessor = TextPreprocessor(
    remove_stopwords=True,
    lemmatize=True
)
```

## 📈 Future Improvements

- [ ] Deep learning models (LSTM, BERT)
- [ ] Real-time sentiment analysis API
- [ ] Multi-language support
- [ ] Aspect-based sentiment analysis
- [ ] Web interface with Streamlit

## 🤝 Contributing

Contributions welcome! Feel free to open issues or submit PRs.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

Kyle Kaufman - [GitHub](https://github.com/kkaufma72)

Project Link: [https://github.com/kkaufma72/Social-Media-Sentiment-Analysis](https://github.com/kkaufma72/Social-Media-Sentiment-Analysis)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NLTK team for NLP tools
- scikit-learn for ML framework
- Open source community
