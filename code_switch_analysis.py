! pip install pandas
! pip install praat-parselmouth librosa
! pip install torchaudio
! pip install transformers
! pip install pydub
! pip install textgrid
! pip install syllables

#Importing necessary libraries
import glob
import os
import librosa
from google.colab import drive
import pandas as pd
import langid
import textgrid
from langdetect import detect
import re
import pyphen
import syllables
import numpy as np
from scipy.io import wavfile
import parselmouth
from parselmouth.praat import call
from IPython.display import Audio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
drive.mount('/content/drive')

#Reading in csv with file, utterances, and utterance time stamps.
df = pd.read_csv('/content/drive/MyDrive/dataverse_files/Spice_Data.csv')

words_df = pd.DataFrame(columns=['file', 'utterance', 'word', 'word_start', 'word_end'])
phones_df = pd.DataFrame(columns=['file', 'utterance', 'phone', 'phone_start', 'phone_end'])

df = df[df['language'] == "English"]

#Defining vowels for later analysis
vowels = ["ER0", "IH2", "EH1", "AE0", "UH1", "AY2", "AW2", "UW1", "OY2", "OY1", "AO0", "AH2", "ER1", "AW1",
          "OW0", "IY1", "IY2", "UW0", "AA1", "EY0", "AE1", "AA0", "OW1", "AW0", "AO1", "AO2", "IH0", "ER2",
          "UW2", "IY0", "AE2", "AH0", "AH1", "UH2", "EH2", "UH0", "EY1", "AY0", "AY1", "EH0", "EY2", "AA2",
          "OW2", "IH1"]

#Going through SpiCE Dataframe with utterances onsets and offsets
for index, row in df.iterrows():
  print(row['utterance'], row['file'])
  #Load in TextGrid for examination
  tg_file = '/content/drive/MyDrive/dataverse_files/english/'+ row['file'] + ".TextGrid"
  tg = textgrid.TextGrid.fromFile(tg_file)

  #Iterate through textgrids
  for tier in tg:
      if tier.name == "word":
        for interval in tier.intervals:
          if (interval.minTime > row['utterance_onset']) & (interval.minTime < row['utterance_offset']):
            pass
          else:
            continue
          word = interval.mark
          df_word = {'file': row['file'], 'utterance': row['utterance'], 'word': word,
                   'word_start': interval.minTime, 'word_end': interval.maxTime, 'cs_next': False}
          words_df = words_df.append(df_word, ignore_index = True)
      elif tier.name == "phone":
        for interval in tier.intervals:
          if interval.minTime >= row['utterance_onset'] and interval.minTime <= row['utterance_offset']:
            pass
          else:
            continue
          df_phone = {'file': row['file'], 'utterance': row['utterance'], 'phone': interval.mark,
                   'phone_start': interval.minTime, 'phone_end': interval.maxTime}
          phones_df = phones_df.append(df_phone, ignore_index = True)

words_df = words_df.dropna()
phones_df = phones_df.dropna()

phones_df.to_csv('/content/drive/MyDrive/dataverse_files/phoneme_data.csv')
words_df.to_csv('/content/drive/MyDrive/dataverse_files/word_data.csv')


#So now we have two DataFrames with the Timestamps of each Phoneme and words, let combines them

words_df = pd.read_csv('/content/drive/MyDrive/dataverse_files/word_data.csv')
phones_df = pd.read_csv('/content/drive/MyDrive/dataverse_files/phoneme_data.csv')
words_df = words_df.dropna()
words_df = words_df.reset_index()

#Coding if the following character is a Chinese character (i.e. before a code-switch)
for index, row in words_df.iterrows():
  if index == len(words_df) - 1:
    break
  else:
    next_word = words_df.at[index + 1, 'word']
    print(row['word'], next_word)
    if next_word == "<unk>":
      words_df.at[index, 'cs_next'] = True
  if row['word']  == "<unk>":
    words_df.drop(index, inplace=True)

words_df['syll/dur'] = ''
for index, row in words_df.iterrows():
  dic = pyphen.Pyphen(lang='en')
  parsed = dic.inserted(row['word'])
  count = syllables.estimate(row['word'])
  dur = row['word_end'] - row['word_start']
   for x in parsed:
     if x == "-":
       count = count + 1
  words_df.at[index, 'syll/dur'] = count / dur

words_df.to_csv('/content/drive/MyDrive/dataverse_files/speech_rate.csv')

phones_df['word'] = ''
phones_df['word_start'] = ''
phones_df['word_end'] = ''
phones_df['cs_next'] = ''

phones_df = phones_df[phones_df['phone'].isin(analysis)]

def in_range(n, start, end = 0):
  return start <= n <= end if end >= start else end <= n <= start

#Matching phones with words based on file, and time stamps.
for index, row in phones_df.iterrows():
  phone = row['phone']
  start = row['phone_start']
  filename = row['file']
  utterance = row['utterance']
  for word_index, word_row in df.iterrows():
    if filename == word_row['file'] and in_range(start, word_row['word_start'], word_row['word_end']):
      phones_df.at[index, 'word_start'] = word_row['word_start']
      phones_df.at[index, 'word_end'] = word_row['word_end']
      phones_df.at[index, 'word'] = word_row['word']
      phones_df.at[index, 'cs_next'] = word_row['cs_next']
    else:
      pass

phones_df.to_csv('/content/drive/MyDrive/dataverse_files/phoneme_and_word_data.csv')

phones_df = pd.read_csv('/content/drive/MyDrive/dataverse_files/phoneme_and_word_data.csv')

phones_df = phones_df.dropna()

def find_middle(lst):
    if not lst:  # Check if the list is empty
        return "The list is empty."

    length = len(lst)  # Get the length of the list

    if length % 2 != 0:  # Check if the length is odd
        middle_index = length // 2
        return lst[middle_index]

    # If the length is even
    first_middle_index = length // 2 - 1
    second_middle_index = length // 2
    return ((lst[first_middle_index] + lst[second_middle_index]) // 2)

def formants_praat(x, start, end):
    ```
    Extracting formants function
    ```
        f0min, f0max = 75, 300
        sound = parselmouth.Sound(x) # read the sound
        sound = sound.extract_part(from_time = start, to_time = end)
        formants = sound.to_formant_burg(time_step=0.010, maximum_formant=5000)

        f1_list, f2_list, f3_list, f4_list  = [], [], [], []
        for t in formants.ts():
            f1 = formants.get_value_at_time(1, t)
            f2 = formants.get_value_at_time(2, t)
            f3 = formants.get_value_at_time(3, t)
            f4 = formants.get_value_at_time(4, t)
            if np.isnan(f1): f1 = 0
            if np.isnan(f2): f2 = 0
            if np.isnan(f3): f3 = 0
            if np.isnan(f4): f4 = 0
            f1_list.append(f1)
            f2_list.append(f2)
            f3_list.append(f3)
            f4_list.append(f4)

        return f1_list, f2_list, f3_list, f4_list

phones_analysis = phones_df[phones_df['phone'].isin(vowels)]
phones_analysis['F1'] = ""
phones_analysis['F2'] = ""


#Adding in F1 and F2 to phone df
for index, row in phones_analysis.iterrows():
  sound = "/content/drive/MyDrive/dataverse_files/english/" + row['file'] + ".wav"
  print(row['utterance'])
  start = row['phone_start']
  end = row['phone_end']
  f0, f1, f2, f3, f4 = formants_praat(sound, start, end)
  f1 = find_middle(f1)
  f2 = find_middle(f2)
  phones_analysis.at[index,'F1'] = f1
  phones_analysis.at[index,'F2'] = f2


phones_analysis.to_csv('/content/drive/MyDrive/dataverse_files/phoneme_analysis_sample_data.csv')

phones_analysis = pd.read_csv('/content/drive/MyDrive/dataverse_files/phoneme_analysis_sample_data.csv')


def barkify (data, formants):
    # For each formant listed, make a copy of the column prefixed with z
    for formant in formants:
        for ch in formant:
            if ch.isnumeric():
                num = ch
        formantchar = (formant.split(num)[0])
        name = str(formant).replace(formantchar,'z')
        # Convert each value from Hz to Bark
        data[name] = 26.81/ (1+ 1960/data[formant]) - 0.53
    # Return the dataframe with the changes
    return data

def Lobify (data, group, formants):
    zscore = lambda x: (x - x.mean()) / x.std()
    for formant in formants:
        name = str("zsc_" + formant)
        col = data.groupby([group])[formant].transform(zscore)
        data.insert(len(data.columns), name, col)
    return data

barked_data = barkify(phones_analysis, ["F1", "F2"])

lobified_data = Lobify(barked_data,
                 group = "file",
                 formants = ["z1", "z2"]
                )

lobified_data.to_csv('/content/drive/MyDrive/dataverse_files/barked_data.csv')

##############################################################################
#Vowel Formant Neural Netowrk                                                #
##############################################################################

df = pd.read_csv('/content/drive/MyDrive/dataverse_files/barked_data.csv')

#Splitting the data for training vs testing
df_cs = df[df['cs_next'] == True]
df_no_cs = df[df['cs_next'] == False]

df_cs_1 = df_cs.iloc[:300,:]
df_cs_2 = df_cs.iloc[300:,:]

df_no_cs_1 = df_no_cs.iloc[:300,:]
df_no_cs_2 = df_no_cs.iloc[300:,:]

df = pd.concat([df_cs_1, df_no_cs_1])
df2 = pd.concat([df_cs_2, df_no_cs_2])

#Separating X and Y variables
label_encoder = LabelEncoder()
df['phone'] = label_encoder.fit_transform(df['phone'])
features = df[['zsc_z1', 'zsc_z2', 'phone']].values
labels = df['cs_next'].astype(int).values

#Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

class VowelFormantPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VowelFormantPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_size = 3  # F1, F2, phone
hidden_size = 64 
output_size = 2  # Before code-switch or not
model = VowelFormantPredictor(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 11

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        # Extract the first 4 columns (F1, F2, stress, phone)
        inputs = batch_X

        outputs = model(inputs)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X_val, batch_y_val in val_loader:
            val_inputs = batch_X_val
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, batch_y_val).item()

    average_val_loss = val_loss / len(val_loader)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}, Val Loss: {average_val_loss:.4f}')

#Moving on to the test data
label_encoder = LabelEncoder()
df2['phone'] = label_encoder.fit_transform(df2['phone'])
features2 = df2[['zsc_z1', 'zsc_z2','phone']].values
labels2 = df2['cs_next'].astype(int).values

#Split the data into training, validation, and test sets
#The test_size parameter determines the percentage of data allocated to the test set
X_train, X_temp, y_train, y_temp = train_test_split(features2, labels2, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#Convert the data to consistent data types
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.long)

#Testing the data
test_data = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model.eval()

predictions = []
ground_truth = []

with torch.no_grad():
    for batch_X_test, batch_y_test in test_loader:
        test_inputs = batch_X_test
        test_outputs = model(test_inputs)

        _, predicted = torch.max(test_outputs, 1)

        predictions.extend(predicted.cpu().numpy())
        ground_truth.extend(batch_y_test.cpu().numpy())

predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

correct_predictions = (predictions == ground_truth).sum()
total_samples = len(ground_truth)
accuracy = correct_predictions / total_samples * 100

print(f'Overall Accuracy: {accuracy:.2f}%')

for i in range(len(predictions)):
    print(f"Example {i + 1}: Prediction={predictions[i]}, Ground Truth={ground_truth[i]}")

##############################################################################
#Speech Rate Neural Network                                                  #
##############################################################################
df = pd.read_csv('/content/drive/MyDrive/dataverse_files/speech_rate.csv')

df_cs = df[df['cs_next'] == True]
df_no_cs = df[df['cs_next'] == False]

df_cs_1 = df_cs.iloc[:300,:]
df_cs_2 = df_cs.iloc[300:,:]

df_no_cs_1 = df_no_cs.iloc[:300,:]
df_no_cs_2 = df_no_cs.iloc[300:,:]

df = pd.concat([df_cs_1, df_no_cs_1])
df2 = pd.concat([df_cs_2, df_no_cs_2])

label_encoder = LabelEncoder()
# df['phone'] = label_encoder.fit_transform(df['phone'])
features = df[['syll/dur']].values
labels = df['cs_next'].astype(int).values

X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

class SRPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VowelFormantPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_size = 1  # Speech rate
hidden_size = 64 
output_size = 2  # Before code-switch or not
model = SRPredictor(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 11

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        # Extract the first 4 columns (F1, F2, stress, phone)
        inputs = batch_X

        outputs = model(inputs)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X_val, batch_y_val in val_loader:
            val_inputs = batch_X_val
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, batch_y_val).item()

    average_val_loss = val_loss / len(val_loader)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}, Val Loss: {average_val_loss:.4f}')

label_encoder = LabelEncoder()
features2 = df2[['syll/dur']].values
labels2 = df2['cs_next'].astype(int).values

X_train, X_temp, y_train, y_temp = train_test_split(features2, labels2, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.long)

test_data = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model.eval()

predictions = []
ground_truth = []

with torch.no_grad():
    for batch_X_test, batch_y_test in test_loader:
        test_inputs = batch_X_test
        test_outputs = model(test_inputs)

        _, predicted = torch.max(test_outputs, 1)

        predictions.extend(predicted.cpu().numpy())
        ground_truth.extend(batch_y_test.cpu().numpy())


predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

correct_predictions = (predictions == ground_truth).sum()
total_samples = len(ground_truth)
accuracy = correct_predictions / total_samples * 100

print(f'Overall Accuracy: {accuracy:.2f}%')

for i in range(len(predictions)):
    print(f"Example {i + 1}: Prediction={predictions[i]}, Ground Truth={ground_truth[i]}")
