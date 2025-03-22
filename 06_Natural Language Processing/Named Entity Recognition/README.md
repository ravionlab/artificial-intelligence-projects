# Named Entity Recognition (NER)

## Overview & Introduction
Named Entity Recognition is a subtask of information extraction that seeks to identify and classify named entities in text into predefined categories such as persons, organizations, locations, medical codes, time expressions, quantities, monetary values, and more.

**Role in Natural Language Processing**:
NER serves as a fundamental component in many NLP pipelines, providing crucial structured information from unstructured text that can be used in downstream tasks like question answering, knowledge graph construction, and information retrieval.

### Historical Context
NER evolved from simple dictionary-based approaches to sophisticated neural models:
- **Early systems (1990s)**: Rule-based and gazetteer approaches
- **Statistical era (2000s)**: Hidden Markov Models, Conditional Random Fields
- **Neural approaches (2010s-present)**: RNNs, Transformers, and contextualized representations

---

## Theoretical Foundations

### Conceptual Explanation
NER treats entity recognition as a sequence labeling problem, where each token in a text sequence is assigned a tag indicating whether it belongs to an entity and what type of entity it represents. The most common tagging scheme is BIO (Beginning, Inside, Outside):
- B-[TYPE]: Beginning of an entity of type [TYPE]
- I-[TYPE]: Inside (continuation) of an entity of type [TYPE]
- O: Outside any entity

### Mathematical Formulation
For a sequence of tokens $X = [x_1, x_2, ..., x_n]$, NER aims to find the most likely sequence of labels $Y = [y_1, y_2, ..., y_n]$ that maximizes:

$$P(Y|X) = \prod_{i=1}^{n} P(y_i | X, y_1, y_2, ..., y_{i-1})$$

In modern neural approaches, this is typically modeled using contextual representations and softmax classifiers.

### Key Approaches
1. **Statistical Models**: Conditional Random Fields (CRFs) that model label dependencies
2. **Recurrent Neural Networks**: Bi-directional LSTMs with CRF layers
3. **Transformer-based Models**: BERT, RoBERTa, etc. with token classification heads
4. **Few-shot Learning**: Prototypical networks and meta-learning for low-resource scenarios

---

## Algorithm Mechanics

### Step-by-Step Process
1. **Text Preprocessing**: Tokenization, sentence splitting
2. **Feature Extraction**: Converting tokens to vector representations
3. **Contextual Encoding**: Capturing contextual information for each token
4. **Tag Prediction**: Classifying each token with the appropriate entity tag
5. **Post-processing**: Ensuring valid tag sequences and entity consistency

### Training & Prediction Workflow
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# Predict named entities
def extract_entities(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted tags
    predictions = torch.argmax(outputs.logits, dim=2)
    
    # Convert IDs to tags and align with original tokens
    predicted_tags = [model.config.id2label[t.item()] for t in predictions[0]]
    word_ids = inputs.word_ids()
    
    # Extract entities (simplified)
    entities = []
    current_entity = None
    
    for idx, tag in enumerate(predicted_tags):
        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"type": tag[2:], "text": tokenizer.decode([inputs.input_ids[0][idx]])}
        elif tag.startswith("I-") and current_entity:
            current_entity["text"] += " " + tokenizer.decode([inputs.input_ids[0][idx]])
        elif tag == "O" and current_entity:
            entities.append(current_entity)
            current_entity = None
    
    return entities

# Example usage
text = "Apple Inc. is planning to open a new store in Berlin next month."
entities = extract_entities(text)
print(entities)
```

---

## Implementation Details

### Code Structure
```python
import torch
import torch.nn as nn
from transformers import BertModel

class BertForNER(nn.Module):
    def __init__(self, num_labels):
        super(BertForNER, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        return logits

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim // 2, 
            bidirectional=True, 
            batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        emissions = self.hidden2tag(lstm_out)
        
        return emissions
        
    def decode(self, x, mask=None):
        emissions = self.forward(x, mask)
        return self.crf.decode(emissions, mask)
```

### Setup Instructions
```bash
# Install required packages
pip install torch transformers seqeval spacy
```

---

## Hyperparameters & Optimization

- **Learning Rate**: Typically in the range of 1e-5 to 5e-5 for fine-tuning pre-trained models
- **Batch Size**: Affects training stability and memory usage
- **Sequence Length**: Maximum number of tokens to process
- **Dropout Rate**: Prevents overfitting
- **Label Smoothing**: Can improve generalization

**Tuning Strategies**:
- Layer-wise learning rate decay for fine-tuning
- Gradient accumulation for larger effective batch sizes
- Mixed precision training for memory efficiency

---

## Evaluation Metrics

- **Entity-level F1 Score**: Harmonic mean of precision and recall at the entity level
- **Token-level F1 Score**: Precision and recall at the token level
- **Exact Match Accuracy**: Percentage of entities exactly matched
- **Partial Match**: Credit for partial entity matches

---

## Practical Examples

**Datasets**:
- CoNLL-2003 (News domain)
- OntoNotes 5.0 (Multi-domain)
- WNUT-17 (Social media)
- BC5CDR (Biomedical)

**Use Cases**:
- Information extraction from documents
- Knowledge graph construction
- Résumé parsing and job matching
- Medical record analysis and clinical NLP
- Legal document analysis

---

## Advanced Theory

**Domain Adaptation**:
Techniques for adapting NER models trained on general domains to specialized ones.

**Nested NER**:
Recognition of entities that contain other entities within them.

**Zero-shot NER**:
Recognizing entity types not seen during training.

**Document-level NER**:
Considering document context for improved entity resolution.

---

## Advantages & Limitations

**Pros**:
- Provides structured information from unstructured text
- Well-studied problem with many available tools
- High performance on standard entity types and domains

**Cons**:
- Domain-specific entities require specialized training data
- Performance degrades on out-of-domain text
- Struggles with ambiguous entities and new entity types
- Challenging for informal text (social media, messaging)

---

## Further Reading

1. Nadeau, D., & Sekine, S. (2007). *A survey of named entity recognition and classification*. Linguisticae Investigationes.
2. Lample, G., et al. (2016). *Neural Architectures for Named Entity Recognition*. NAACL.
3. Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
4. Li, X., et al. (2020). *A Survey on Deep Learning for Named Entity Recognition*. IEEE Transactions on Knowledge and Data Engineering.
5. Lin, B. Y., et al. (2019). *A General Framework for Information Extraction using Dynamic Span Graphs*. NAACL.
