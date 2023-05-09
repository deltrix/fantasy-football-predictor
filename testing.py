import json
import csv

csv.field_size_limit(100000000)

import time

# Import scraping modules
from urllib.request import urlopen
from bs4 import BeautifulSoup

# Import data manipulation modules
import pandas as pd
import numpy as np
import seaborn as sns

# Import data visualization modules
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def readJSON(year):
    passing_stats = []
    passing_headers = []
    rushing_stats = []
    rushing_headers = []
    receiving_stats = []
    receiving_headers = []

    with open('json/pa' + year + 'raw.json', newline='', encoding='utf-8') as file:
        j = json.load(file)
        passing_stats = BeautifulSoup(j)
    with open('json/pa' + year + 'head.json', newline='', encoding='utf-8') as file:
        s = json.load(file)
        j = s[1:-1].split(',')
        j = [s.replace(" ", "").replace("'", "") for s in j]
        passing_headers = j
    with open('json/ru' + year + 'raw.json', newline='', encoding='utf-8') as file:
        j = json.load(file)
        rushing_stats = BeautifulSoup(j)
    with open('json/ru' + year + 'head.json', newline='', encoding='utf-8') as file:
        s = json.load(file)
        j = s[1:-1].split(',')
        j = [s.replace(" ", "").replace("'", "") for s in j]
        rushing_headers = j
    with open('json/re' + year + 'raw.json', newline='', encoding='utf-8') as file:
        j = json.load(file)
        receiving_stats = BeautifulSoup(j)
    with open('json/re' + year + 'head.json', newline='', encoding='utf-8') as file:
        s = json.load(file)
        j = s[1:-1].split(',')
        j = [s.replace(" ", "").replace("'", "") for s in j]
        receiving_headers = j

    return passing_stats, rushing_stats, receiving_stats, passing_headers, rushing_headers, receiving_headers


# Train and evaluate model
def train_and_evaluate_model(model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer,
                             num_epochs=1000, device="cpu"):
    train_losses = []
    val_losses = []

    # Move model to device
    model.to(device)

    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.view(-1, 1)  # reshape the target tensor
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Compute validation loss
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                targets = targets.view(-1, 1)  # reshape the target tensor
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        train_losses.append(train_loss / len(train_dataloader))
        val_losses.append(val_loss / len(val_dataloader))

        print(f"Epoch {epoch + 1} - Train Loss: {train_losses[-1]:.4f} - Val Loss: {val_losses[-1]:.4f}")

    # Evaluate model on test set
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            test_loss += loss.item()

    print(f'Test Loss: {test_loss / len(test_dataloader):.4f}')

    return train_losses, val_losses, test_loss / len(test_dataloader)


def main():
    years = 23

    passing_stats = []
    passing_headers = []
    rushing_stats = []
    rushing_headers = []
    receiving_stats = []
    receiving_headers = []

    for i in range(10):
        pa_stats, ru_stats, re_stats, pa_headers, ru_headers, re_headers = readJSON("200" + str(i))
        passing_stats.append(pa_stats)
        rushing_stats.append(ru_stats)
        receiving_stats.append(re_stats)
        passing_headers.append(pa_headers)
        rushing_headers.append(ru_headers)
        receiving_headers.append(re_headers)
    for i in range(10, 23):
        pa_stats, ru_stats, re_stats, pa_headers, ru_headers, re_headers = readJSON("20" + str(i))
        passing_stats.append(pa_stats)
        rushing_stats.append(ru_stats)
        receiving_stats.append(re_stats)
        passing_headers.append(pa_headers)
        rushing_headers.append(ru_headers)
        receiving_headers.append(re_headers)

    # Table headers and table rows ==================================================
    print("---Table headers and table rows")

    passing = [[] for i in range(years)]
    receiving = [[] for i in range(years)]
    rushing = [[] for i in range(years)]

    for i in range(years):
        rows = passing_stats[i].findAll('tr')[1:]
        passing[i] = []
        for x in range(len(rows)):
            passing[i].append([col.getText() for col in rows[x].findAll('td')])
        rows = receiving_stats[i].findAll('tr')[1:]
        receiving[i] = []
        for x in range(len(rows)):
            receiving[i].append([col.getText() for col in rows[x].findAll('td')])
        rows = rushing_stats[i].findAll('tr')[1:]
        rushing[i] = []
        for x in range(len(rows)):
            rushing[i].append([col.getText() for col in rows[x].findAll('td')])

    # print(passing_headers[22])
    # print(rushing_headers[22])
    # print(receiving_headers[22])
    # all good

    # Create the dataframes ==================================================
    print("---Create the dataframes")

    rushing_data = []
    receiving_data = []
    passing_data = []

    for i in range(years):
        # print("i is " + str(i))

        df_ru = pd.DataFrame(rushing[i], columns=rushing_headers[i][1:])
        columns_to_drop = ["1D", "Lng"]
        columns_existing = [col for col in columns_to_drop if col in df_ru.columns]
        if columns_existing:
            df_ru.drop(columns=columns_existing, axis=1, inplace=True)
        df_ru.rename(
            columns={'Att': 'Rush_Att', 'Yds': 'Rush_Yds', 'Y/A': 'Rush_Y/A', 'Y/G': 'Rush_Y/G', 'TD': 'Rush_TD'},
            inplace=True)
        rushing_data.append(df_ru)

        df_re = pd.DataFrame(receiving[i], columns=receiving_headers[i][1:])
        columns_to_drop = ["1D", "Ctch%", "Lng"]
        columns_existing = [col for col in columns_to_drop if col in df_re.columns]
        if columns_existing:
            df_re.drop(columns=columns_existing, axis=1, inplace=True)
        df_re.rename(columns={'Yds': 'Receiving_Yds', 'Y/G': 'Receiving_Y/G', 'TD': 'Receiving_TD'}, inplace=True)
        receiving_data.append(df_re)

        df_pa = pd.DataFrame(passing[i], columns=passing_headers[i][1:])
        new_cols = df_pa.columns.values
        new_cols[-6] = 'Yds_Sacked'
        df_pa.columns = new_cols
        columns_to_drop = ['QBrec', 'Yds_Sacked', 'Sk%', '4QC', 'GWD', 'NY/A', 'TD%', 'Int%', '1D', 'Y/A', 'Lng', 'QBR',
                           'Cmp%']
        columns_existing = [col for col in columns_to_drop if col in df_pa.columns]
        if columns_existing:
            df_pa.drop(columns=columns_existing, axis=1, inplace=True)
        df_pa.rename(columns={'Yds': 'Passing_Yds', 'Att': 'Pass_Att', 'Y/G': 'Pass_Y/G', 'TD': 'Pass_TD'},
                     inplace=True)
        passing_data.append(df_pa)

        receiving_data[i]['Player'] = receiving_data[i]['Player'].str.replace('*', '', regex=False)
        receiving_data[i]['Player'] = receiving_data[i]['Player'].str.replace('+', '', regex=False)
        rushing_data[i]['Player'] = rushing_data[i]['Player'].str.replace('*', '', regex=False)
        rushing_data[i]['Player'] = rushing_data[i]['Player'].str.replace('+', '', regex=False)
        passing_data[i]['Player'] = passing_data[i]['Player'].str.replace('*', '', regex=False)
        passing_data[i]['Player'] = passing_data[i]['Player'].str.replace('+', '', regex=False)

        rushing_data[i] = rushing_data[i].dropna()
        receiving_data[i] = receiving_data[i].dropna()
        passing_data[i] = passing_data[i].dropna()

    rushing_data[22].head()  # Check if works

    # Merge our dataframes to create one overall dataframe ==================================================
    print("---Merge our dataframes to create one overall dataframe")

    df = [[] for i in range(years)]
    for i in range(years):
        WR_RB_df = pd.merge(rushing_data[i], receiving_data[i], on='Player', how='outer')
        df[i] = pd.merge(WR_RB_df, passing_data[i], on='Player', how='outer')
        # Combine Position to get one column, drop others
        df[i]['Position'] = df[i]['Pos_x'].combine_first(df[i]['Pos_y']).combine_first(df[i]['Pos'])
        df[i].drop(['Pos_x', 'Pos_y', 'Pos'], axis=1, inplace=True)
        df[i]['Team'] = df[i]['Tm_x'].combine_first(df[i]['Tm_y']).combine_first(df[i]['Tm'])
        df[i].drop(['Tm_x', 'Tm_y', 'Tm'], axis=1, inplace=True)
        df[i]['Fumbles'] = df[i]['Fmb_x'].combine_first(df[i]['Fmb_y'])
        df[i].drop(['Fmb_x', 'Fmb_y'], axis=1, inplace=True)
        df[i]['Games_Played'] = df[i]['G_x'].combine_first(df[i]['G_y']).combine_first(df[i]['G'])
        df[i].drop(['G_x', 'G_y', 'G'], axis=1, inplace=True)
        df[i]['Games_Started'] = df[i]['GS_x'].combine_first(df[i]['GS_y']).combine_first(df[i]['GS'])
        df[i].drop(['GS_x', 'GS_y', 'GS'], axis=1, inplace=True)
        df[i]['Age'] = df[i]['Age_x'].combine_first(df[i]['Age_y'])
        df[i].drop(['Age_x', 'Age_y'], axis=1, inplace=True)

    # Create categories to convert types ==================================================
    print("---Create categories to convert types")

    categories = ['Age', 'Rush_Att', 'Rush_Yds', 'Rush_TD', 'Rush_Y/A', 'Rush_Y/G', 'Tgt', 'Rec', 'Receiving_Yds',
                  'Y/R', 'Receiving_TD', 'Y/Tgt', 'R/G', 'Receiving_Y/G', 'Cmp', 'Pass_Att', 'Passing_Yds', 'Pass_TD',
                  'Int', 'AY/A', 'Y/C', 'Pass_Y/G', 'Rate', 'Sk', 'ANY/A', 'Fumbles', 'Games_Played', 'Games_Started']

    # convert data types to numeric
    for i in range(years):
        for x in categories:
            df[i][x] = pd.to_numeric(df[i][x])
        df[i] = df[i].fillna(0)

    # Touchdowns and Fantasy Points ==================================================
    print("---Touchdowns and Fantasy Points")

    # Add up touchdowns as they are worth the same amount of points
    for i in range(years):
        df[i]['Touchdowns'] = df[i]['Pass_TD'] + df[i]['Rush_TD'] + df[i]['Receiving_TD']
        df[i]['Fantasy_Points'] = (
                df[i]['Passing_Yds'] / 25 +
                df[i]['Pass_TD'] * 4 +
                df[i]['Int'] * -2 +
                df[i]['Rush_Yds'] / 10 +
                df[i]['Rush_TD'] * 6 +
                df[i]['Rec'] +
                df[i]['Receiving_Yds'] / 10 +
                df[i]['Receiving_TD'] * 6 +
                df[i]['Fumbles'] * -2
        )

    # ==================================================
    # NEURAL NETWORK TESTING ==================================================
    # ==================================================
    print("---NEURAL NETWORK TESTING")

    chosen_year = 21

    choice = 'QB'
    if choice == 'RB':
        X = df[chosen_year][
            ['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Fumbles',
             'Rush_Att']]
        y = df[chosen_year]['Fantasy_Points']
    elif choice == 'WR':
        X = df[chosen_year][
            ['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Tgt', 'Rec',
             'Fumbles']]
        y = df[chosen_year]['Fantasy_Points']
    elif choice == 'QB':
        X = df[chosen_year][
            ['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Int', 'Sk',
             'Rate']]
        y = df[chosen_year]['Fantasy_Points']

    # Split data into training, validation, and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val.values, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).to(device)

    # Define dataloader objects
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    # Define neural network architecture
    class TwoLayerMLP(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, dropout_prob):
            super(TwoLayerMLP, self).__init__()

            self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
            self.dropout = torch.nn.Dropout(p=dropout_prob)
            self.layer2 = torch.nn.Linear(hidden_dim, 1)

        def forward(self, d):
            d = F.relu(self.layer1(d))
            d = self.dropout(d)
            d = self.layer2(d)
            return d

    # Define model hyper parameters
    input_dim = X_train.shape[1]
    hidden_dim = 64
    dropout_prob = 0.1
    lr = 0.001
    num_epochs = 100
    patience = 10

    model = TwoLayerMLP(input_dim, hidden_dim, dropout_prob).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train and evaluate model
    train_losses, val_losses, test_loss = train_and_evaluate_model(model, train_dataloader, val_dataloader,
                                                                   test_dataloader, criterion, optimizer,
                                                                   num_epochs=100, device=device)

    # Plot training and validation losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.show()

    # Evaluate model on test set
    with torch.no_grad():
        test_loss = 0.0
        for inputs, targets in test_dataloader:
            outputs = model(inputs)
            targets = targets.view(-1, 1)  # reshape the target tensor
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    print(f'Test Loss: {test_loss / len(test_dataloader):.4f}')


if __name__ == '__main__':
    main()
