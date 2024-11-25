import csv
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import set_seed
import torch
import torch.nn as nn
import warnings
import pandas as pd

warnings.filterwarnings('ignore')
set_seed(4)
device = "cuda:0"
checkpoint = './models--facebook--esm2_t30_150M_UR50D'
AutoTokenizer = AutoTokenizer.from_pretrained(checkpoint)


def AFPpred(file):
    test_sequences = file
    max_len = 30
    test_data = AutoTokenizer(test_sequences, max_length=max_len, padding="max_length", truncation=True,
                              return_tensors='pt')

    class ESModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=320)
            self.bn1 = nn.BatchNorm1d(256)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(64)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(320, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 64)
            self.output_layer = nn.Linear(64, 2)
            self.dropout = nn.Dropout(0)

        def forward(self, x):
            with torch.no_grad():
                bert_output = self.bert(input_ids=x['input_ids'].to(device),
                                        attention_mask=x['attention_mask'].to(device))
            output_feature = self.dropout(bert_output["logits"])
            output_feature = self.relu(self.bn1(self.fc1(output_feature)))
            output_feature = self.relu(self.bn2(self.fc2(output_feature)))
            output_feature = self.relu(self.bn3(self.fc3(output_feature)))
            output_feature = self.output_layer(output_feature)
            return torch.softmax(output_feature, dim=1)

    model = ESModel()
    model.load_state_dict(torch.load("./models/train_best_model.pth"))
    model = model.to(device)
    model.eval()
    out_probability = []
    with torch.no_grad():
        predict = model(test_data)
        out_probability.extend(np.max(np.array(predict.cpu()), axis=1).tolist())
        test_argmax = np.argmax(predict.cpu(), axis=1).tolist()

    id2str = {0: "non-AFP", 1: "AFP"}
    predicted_label = id2str[test_argmax[0]]
    predicted_probability = out_probability[0]
    return predicted_label, predicted_probability

# Read the CSV file of your saved sequence
use_cols = ['Seq']  # Label
data = pd.read_csv('./data/input.csv', usecols=use_cols)  # csv reading and keywords
data_list = data.to_dict('records')

# Files used to store results
filename = './data/output.csv'

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Seq', 'Result'])

    for peptide in data_list:
        predicted = AFPpred(peptide['Seq'])
        writer.writerow([peptide['Seq'], predicted])

print("The data has been successfully written to the file:", filename)