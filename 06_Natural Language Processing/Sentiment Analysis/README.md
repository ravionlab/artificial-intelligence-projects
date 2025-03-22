# Sentiment Analysis

## Overview & Introduction
Sentiment Analysis is a natural language processing technique that identifies and extracts subjective information from text to determine the writer's attitude, opinions, or emotions toward a particular topic, product, service, or experience.

**Role in Natural Language Processing**:
Sentiment analysis serves as a crucial application of NLP that enables organizations to monitor brand reputation, understand customer feedback, analyze market trends, and make data-driven decisions based on public opinion.

### Historical Context
Sentiment analysis has evolved through several distinct phases:
- **Lexicon-based approaches (Early 2000s)**: Used dictionaries of words labeled with sentiment polarities
- **Machine learning classifiers (2000s-2010s)**: Feature engineering with SVM, Naive Bayes, etc.
- **Deep learning era (2010s-present)**: Advanced neural architectures capturing semantic nuances

---

## Theoretical Foundations

### Conceptual Explanation
At its core, sentiment analysis involves classifying text into sentiment categories, typically:
- Positive: Expressing favorable opinions or emotions
- Negative: Expressing unfavorable opinions or emotions
- Neutral: Factual or objective without clear sentiment

More advanced systems may include:
- Fine-grained sentiment (very positive to very negative)
- Emotion detection (joy, anger, sadness, fear, etc.)
- Aspect-based sentiment (analyzing sentiment toward specific aspects)

### Mathematical Formulation
For a document or text snippet $D$ consisting of tokens $[w_1, w_2, ..., w_n]$, sentiment analysis aims to find a mapping function $f$ such that:

$$f(D) = S$$

Where $S$ represents the sentiment class or score. In probabilistic terms, we seek to maximize:

$$P(S|D) = P(S|w_1, w_2, ..., w_n)$$

### Key Approaches
1. **Lexicon-based Methods**: Using sentiment dictionaries and linguistic rules
2. **Traditional ML**: Naive Bayes, SVM, or Random Forest with engineered features
3. **Deep Learning**: CNNs, RNNs, and Transformer-based architectures
4. **Transfer Learning**: Fine-tuning pre-trained language models for sentiment tasks

---

## Algorithm Mechanics

### Step-by-Step Process
1. **Text Preprocessing**: Tokenization, normalization, removing stopwords
2. **Feature Extraction**: Converting text to numerical features
   - Bag-of-words, TF-IDF, word embeddings, contextual embeddings
3. **Classification/Regression**: Applying the model to determine sentiment
4. **Post-processing**: Interpreting outputs, applying thresholds, etc.

### Training & Prediction Workflow
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Analyze sentiment
def analyze_sentiment(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        scores = F.softmax(outputs.logits, dim=1)
        
    # Extract results
    labels = ["negative", "positive"]
    result = {
        "text": text,
        "sentiment": labels[scores.argmax().item()],
        "confidence": scores.max().item(),
        "scores": {label: score.item() for label, score in zip(labels, scores[0])}
    }
    
    return result

# Example usage
review = "The movie was fantastic! Great performances and an engaging plot."
sentiment = analyze_sentiment(review)
print(sentiment)
```

---

## Implementation Details

### Code Structure
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, 
                      kernel_size=(fs, embedding_dim)) 
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text = [batch size, sentence length]
        embedded = self.embedding(text)
        # embedded = [batch size, sentence length, embedding dim]
        
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sentence length, embedding dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, num_filters, sentence length - filter_sizes[n] + 1]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, num_filters]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, num_filters * len(filter_sizes)]
        
        return self.fc(cat)

class LSTM_SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text = [batch size, sentence length]
        embedded = self.embedding(text)
        # embedded = [batch size, sentence length, embedding dim]
        
        output, (hidden, cell) = self.lstm(embedded)
        # output = [batch size, sentence length, hidden dim * num directions]
        # hidden = [num layers * num directions, batch size, hidden dim]
        
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        # hidden = [batch size, hidden dim * num directions]
            
        return self.fc(self.dropout(hidden))
```

### Setup Instructions
```bash
# Install required packages
pip install torch transformers nltk scikit-learn vaderSentiment textblob
```

---

## Hyperparameters & Optimization

- **Learning Rate**: Typically 2e-5 to 5e-5 for fine-tuning pre-trained models
- **Batch Size**: Usually 16-32 for transformer models
- **Sequence Length**: Maximum text length (often 128 or 512 tokens)
- **Embedding Dimension**: Size of word vectors (300-768 common)
- **Dropout Rate**: Prevents overfitting (0.1-0.5 typical)

**Tuning Strategies**:
- Early stopping based on validation performance
- Learning rate scheduling (linear decay, cosine annealing)
- Data augmentation techniques (back-translation, synonym replacement)

---

## Evaluation Metrics

- **Accuracy**: Proportion of correctly classified instances
- **Precision, Recall, F1-Score**: Especially useful for imbalanced datasets
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve
- **Macro vs. Micro Averaging**: For multi-class sentiment analysis
- **Mean Absolute Error**: For regression-based sentiment intensity

---

## Practical Examples

**Datasets**:
- SST (Stanford Sentiment Treebank)
- IMDB Movie Reviews
- Amazon Product Reviews
- Twitter Sentiment Analysis
- SemEval competition datasets

**Use Cases**:
- Product review analysis
- Social media monitoring
- Brand reputation management
- Market research and competitive analysis
- Political opinion tracking
- Customer feedback analysis

---

## Advanced Theory

**Aspect-Based Sentiment Analysis (ABSA)**:
Identifying sentiment towards specific aspects/features of products or services.

**Cross-Domain Sentiment Analysis**:
Transferring knowledge between domains with different vocabulary and expressions.

**Multimodal Sentiment Analysis**:
Combining text with other modalities (voice, facial expressions, images).

**Contextual Polarity Disambiguation**:
Handling negation, sarcasm, and implicit sentiment.

---

## Advantages & Limitations

**Pros**:
- Automated analysis of large volumes of textual data
- Real-time monitoring of public opinion
- Quantifiable measure of subjective information
- Consistent evaluation method compared to human analysts

**Cons**:
- Difficulty with sarcasm, irony, and humor
- Challenges with implicit sentiment and figurative language
- Cultural and contextual sensitivity
- Domain dependency requiring adaptation
- Language-specific considerations for multilingual analysis

---

## Further Reading

1. Liu, B. (2020). *Sentiment Analysis: Mining Opinions, Sentiments, and Emotions*. Cambridge University Press.
2. Mohammad, S. M. (2022). *Practical Natural Language Processing: A Comprehensive Guide to Building Real-World NLP Systems*. O'Reilly Media.
3. Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
4. Socher, R., et al. (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank*. EMNLP.
5. Zhang, L., Wang, S., & Liu, B. (2018). *Deep Learning for Sentiment Analysis: A Survey*. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery.
