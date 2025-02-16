import torch
import json
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer

# Load the fine-tuned model and tokenizer
model = MobileBertForSequenceClassification.from_pretrained(".")
tokenizer = MobileBertTokenizer.from_pretrained(".")
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
