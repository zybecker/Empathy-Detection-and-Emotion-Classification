#!/usr/bin/env python
# coding: utf-8


import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# get rid of the rows with data outside the first 12 columns, skip rows with parsing errors
train_data = pd.read_csv('trac2_CONVT_train.csv', usecols=range(12), on_bad_lines='skip') 
dev_data = pd.read_csv('trac2_CONVT_dev.csv', usecols=range(12), on_bad_lines='skip')

# Convert emotion, emotional polarity, empathy to numeric vals, replace invalid values with NaNs
for col in ['Emotion', 'EmotionalPolarity', 'Empathy']:
    train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
    dev_data[col] = pd.to_numeric(dev_data[col], errors='coerce')

train_data = train_data.dropna(subset=['Emotion', 'EmotionalPolarity', 'Empathy']).reset_index(drop=True)
dev_data = dev_data.dropna(subset=['Emotion', 'EmotionalPolarity', 'Empathy']).reset_index(drop=True)

class ConversationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):   #extract the text, emotion, emotion polarity, empathy labels
        text = self.data.iloc[index]['text']
        labels = self.data.iloc[index][['Emotion', 'EmotionalPolarity', 'Empathy']].values.astype(float)
        encoding = self.tokenizer(text,
                                  max_length=self.max_length,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')

        # tokenize text into input IDs and attention masks
        return {'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(labels, dtype=torch.float)}

# Initialize BERT tokenizer, convert train/validation data into PyTorch dataset objects
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = ConversationDataset(train_data, tokenizer)
dev_dataset = ConversationDataset(dev_data, tokenizer)

# wrap datasets into DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

#model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.MSELoss() # MSE as the loss function

#training
for epoch in range(10):  
    model.train()
    train_loss = 0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        #calculate forward pass and compute loss then backprop the gradients and update params
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
    print(f"Epoch {epoch + 1}: Train Loss = {train_loss / len(train_loader):}")

#validating
model.eval()
predictions = []
val_labels = []
with torch.no_grad():
    for batch in dev_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions.append(outputs.logits.cpu().numpy())
        val_labels.append(labels.cpu().numpy())

#flatten predictions and labels into arrays
predictions = np.concatenate(predictions, axis=0)
val_labels = np.concatenate(val_labels, axis=0)

#compute scores
pearson_emotion = pearsonr(val_labels[:, 0], predictions[:, 0])[0]
pearson_emotional_polarity = pearsonr(val_labels[:, 1], predictions[:, 1])[0]
pearson_empathy = pearsonr(val_labels[:, 2], predictions[:, 2])[0]
average_pearson = (pearson_emotion + pearson_emotional_polarity + pearson_empathy)/3

print("Emotion Intensity Pearson Score:", pearson_emotion)
print("Emotional Polarity Pearson Score:", pearson_emotional_polarity)
print("Empathy Pearson Score:", pearson_empathy)
print("Average Pearson Score:", average_pearson)



# Examine the ranges and distributions of the labels
print(train_data[['Emotion', 'EmotionalPolarity', 'Empathy']].describe())


#apply to the goldstandard test data
test_data = pd.read_csv('goldstandard_CONVT.csv', on_bad_lines='skip')

for col in ['Emotion', 'EmotionalPolarity', 'Empathy']:
    test_data[col] = pd.to_numeric(test_data[col], errors='coerce')

test_data = test_data.dropna(subset=['Emotion', 'EmotionalPolarity', 'Empathy']).reset_index(drop=True)

test_dataset = ConversationDataset(test_data, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model.eval()
test_predictions = []
test_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        test_predictions.append(outputs.logits.cpu().numpy())
        test_labels.append(labels.cpu().numpy())

test_predictions = np.concatenate(test_predictions, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

test_pearson_emotion = pearsonr(test_labels[:, 0], test_predictions[:, 0])[0]
test_pearson_emotional_polarity = pearsonr(test_labels[:, 1], test_predictions[:, 1])[0]
test_pearson_empathy = pearsonr(test_labels[:, 2], test_predictions[:, 2])[0]
test_average_pearson = (test_pearson_emotion + test_pearson_emotional_polarity + test_pearson_empathy) / 3

print("Emotion Intensity Pearson Score:", test_pearson_emotion)
print("Emotional Polarity Pearson Score:", test_pearson_emotional_polarity)
print("Empathy Pearson Score:", test_pearson_empathy)
print("Average Pearson Score:", test_average_pearson)

