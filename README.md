# MobileBert-Luna: Fine-Tuned MobileBERT for Intent Classification

## Overview

**MobileBert-Luna** is a fine-tuned version of Google's [MobileBERT-uncased](https://huggingface.co/google/mobilebert-uncased) designed for intent classification tasks. This model has been trained on a subset of the [MASSIVE dataset](https://huggingface.co/datasets/massive), a large, multilingual corpus containing over 1 million utterances with annotations for intent prediction and slot annotation.

## Model Details

MobileBert-Luna leverages the efficiency and compactness of MobileBERT, adapting it for robust intent classification. Key aspects include:

- **Base Model:**  
  [MobileBERT-uncased](https://huggingface.co/google/mobilebert-uncased) by Google.

- **Dataset:**  
  [MASSIVE](https://huggingface.co/datasets/qanastek/MASSIVE) – A parallel dataset with over 1M utterances across 51 languages, annotated for natural language understanding tasks including intent prediction.

- **Task:**  
  Fine-tuning was performed for intent classification, mapping user utterances to one of 60 possible intents from Alexa-Intents-Classfier(e.g., `alarm_set`, `play_music`, `calendar_query`, etc.).

- **Training:**  
  The model was trained using a sequence classification head on MobileBERT. A custom label mapping was applied to convert between numeric class indices and human-readable intent names.

## Intended Use

MobileBert-Luna is ideal for applications that require real-time, multilingual natural language understanding, such as:
- Intelligent voice assistants
- Chatbots
- Automated customer service systems
- Any application that benefits from accurate intent detection

## How to Use

Below is an example of how to load and use MobileBert-Luna with the Hugging Face Transformers library:

```python
import torch
import json
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer

# Load the fine-tuned model and tokenizer
model = MobileBertForSequenceClassification.from_pretrained("path/to/mobilebert-luna")
tokenizer = MobileBertTokenizer.from_pretrained("path/to/mobilebert-luna-tokenizer")
model.eval()

# Load the intent mapping from the external JSON file
with open("intent_mapping.json", "r") as f:
    intent_mapping = json.load(f)
# Convert keys to integers if they were saved as strings in JSON
intent_mapping = {int(k): v for k, v in intent_mapping.items()}

# Sample inference
sample_text = "Wake me up at nine am on friday"
inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)

if torch.cuda.is_available():
    model.to("cuda")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    predicted_id = torch.argmax(outputs.logits, dim=-1).item()

predicted_intent = intent_mapping.get(predicted_id, "Unknown Intent")

print("Predicted Intent ID:", predicted_id)
print("Predicted Intent Name:", predicted_intent)
```
