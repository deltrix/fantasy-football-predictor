import argparse
import sys
import os

import json
# import csv
# csv.field_size_limit(100000000)

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

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# parse data from pro-football-reference with BeautifulSoup4
def parse_stats(year):

    passing_url = 'https://www.pro-football-reference.com/years/' + year +'/passing.htm'
    rushing_url = 'https://www.pro-football-reference.com/years/' + year +'/rushing.htm'
    receiving_url = 'https://www.pro-football-reference.com/years/' + year +'/receiving.htm'

    # Set sleep 5 seconds since error 429 occured, expect 15s * 23 = ~5min to run
    wait = 1
    html = urlopen(passing_url)
    passing_stats = BeautifulSoup(html)
    time.sleep(wait)
    html = urlopen(rushing_url)
    rushing_stats = BeautifulSoup(html)
    time.sleep(wait)
    html = urlopen(receiving_url)
    receiving_stats = BeautifulSoup(html)
    time.sleep(wait)

    passing_col_headers = passing_stats.findAll('tr')[0]
    passing_col_headers = [i.getText() for i in passing_col_headers.findAll('th')]
    rushing_col_headers = rushing_stats.findAll('tr')[1]
    rushing_col_headers = [i.getText() for i in rushing_col_headers.findAll('th')]
    receiving_col_headers = receiving_stats.findAll('tr')[0]
    receiving_col_headers = [i.getText() for i in receiving_col_headers.findAll('th')]

    return passing_stats, rushing_stats, receiving_stats, passing_col_headers, rushing_col_headers, receiving_col_headers

# read the data from the csv files
# realized that csv doesnt work well with BeautifulSoup objects
# def readCSV(year):

#     passing_stats = []
#     passing_headers = []
#     rushing_stats = []
#     rushing_headers = []
#     receiving_stats = []
#     receiving_headers = []

#     with open('csv/pa' + year + 'raw.csv', newline='', encoding='utf-8') as csvfile:
#         reader = csv.reader(csvfile)
#         for row in reader:
#             passing_stats.append(row)
#     with open('csv/pa' + year + 'head.csv', newline='', encoding='utf-8') as csvfile:
#         reader = csv.reader(csvfile)
#         for row in reader:
#             passing_headers = row
#     with open('csv/ru' + year + 'raw.csv', newline='', encoding='utf-8') as csvfile:
#         reader = csv.reader(csvfile)
#         for row in reader:
#             rushing_stats.append(row)
#     with open('csv/ru' + year + 'head.csv', newline='', encoding='utf-8') as csvfile:
#         reader = csv.reader(csvfile)
#         for row in reader:
#             rushing_headers = row
#     with open('csv/re' + year + 'raw.csv', newline='', encoding='utf-8') as csvfile:
#         reader = csv.reader(csvfile)
#         for row in reader:
#             receiving_stats.append(row)
#     with open('csv/re' + year + 'head.csv', newline='', encoding='utf-8') as csvfile:
#         reader = csv.reader(csvfile)
#         for row in reader:
#             receiving_headers = row


#    return passing_stats, rushing_stats, receiving_stats, passing_headers, rushing_headers, receiving_headers

def readJSON(year):

    passing_stats = []
    passing_headers = []
    rushing_stats = []
    rushing_headers = []
    receiving_stats = []
    receiving_headers = []

    with open('json/pa' + year + 'raw.json', newline='', encoding='utf-8') as file:
        j = json.load(file)
        passing_stats = BeautifulSoup(j, features="lxml")
    with open('json/pa' + year + 'head.json', newline='', encoding='utf-8') as file:
        s = json.load(file)
        j = s[1:-1].split(',')
        j = [s.replace(" ", "").replace("'", "") for s in j]
        passing_headers = j
    with open('json/ru' + year + 'raw.json', newline='', encoding='utf-8') as file:
        j = json.load(file)
        rushing_stats = BeautifulSoup(j, features="lxml")
    with open('json/ru' + year + 'head.json', newline='', encoding='utf-8') as file:
        s = json.load(file)
        j = s[1:-1].split(',')
        j = [s.replace(" ", "").replace("'", "") for s in j]
        rushing_headers = j
    with open('json/re' + year + 'raw.json', newline='', encoding='utf-8') as file:
        j = json.load(file)
        receiving_stats = BeautifulSoup(j, features="lxml")
    with open('json/re' + year + 'head.json', newline='', encoding='utf-8') as file:
        s = json.load(file)
        j = s[1:-1].split(',')
        j = [s.replace(" ", "").replace("'", "") for s in j]
        receiving_headers = j


    return passing_stats, rushing_stats, receiving_stats, passing_headers, rushing_headers, receiving_headers



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

    #print(passing_headers[22])
    #print(rushing_headers[22])
    #print(receiving_headers[22])
    # all good

    # Create the dataframes ==================================================
    print("---Create the dataframes")

    rushing_data = []
    receiving_data= []
    passing_data = []

    for i in range(years):
        #print("i is " + str(i))
        
        df_ru = pd.DataFrame(rushing[i], columns = rushing_headers[i][1:])
        columns_to_drop = ["1D", "Lng"]
        columns_existing = [col for col in columns_to_drop if col in df_ru.columns]
        if columns_existing:
            df_ru.drop(columns=columns_existing, axis=1, inplace=True)
        df_ru.rename(columns={'Att':'Rush_Att', 'Yds':'Rush_Yds', 'Y/A':'Rush_Y/A', 'Y/G':'Rush_Y/G', 'TD':'Rush_TD'}, inplace=True)
        rushing_data.append(df_ru)

        df_re = pd.DataFrame(receiving[i], columns = receiving_headers[i][1:])
        columns_to_drop = ["1D", "Ctch%", "Lng"]
        columns_existing = [col for col in columns_to_drop if col in df_re.columns]
        if columns_existing:
            df_re.drop(columns=columns_existing, axis=1, inplace=True)
        df_re.rename(columns={'Yds':'Receiving_Yds', 'Y/G':'Receiving_Y/G', 'TD':'Receiving_TD'}, inplace=True)
        receiving_data.append(df_re)

        df_pa = pd.DataFrame(passing[i], columns = passing_headers[i][1:])
        new_cols = df_pa.columns.values
        new_cols[-6] = 'Yds_Sacked'
        df_pa.columns = new_cols
        columns_to_drop = ['QBrec','Yds_Sacked', 'Sk%', '4QC', 'GWD', 'NY/A', 'TD%', 'Int%', '1D', 'Y/A', 'Lng', 'QBR', 'Cmp%']
        columns_existing = [col for col in columns_to_drop if col in df_pa.columns]
        if columns_existing:
            df_pa.drop(columns=columns_existing, axis=1, inplace=True)
        df_pa.rename(columns={'Yds':'Passing_Yds', 'Att':'Pass_Att', 'Y/G':'Pass_Y/G', 'TD':'Pass_TD'}, inplace=True)
        passing_data.append(df_pa)

        receiving_data[i]['Player'] = receiving_data[i]['Player'].str.replace('*', '', regex = False)
        receiving_data[i]['Player'] = receiving_data[i]['Player'].str.replace('+', '', regex = False)
        rushing_data[i]['Player'] = rushing_data[i]['Player'].str.replace('*', '', regex = False)
        rushing_data[i]['Player'] = rushing_data[i]['Player'].str.replace('+', '', regex = False)
        passing_data[i]['Player'] = passing_data[i]['Player'].str.replace('*', '', regex = False)
        passing_data[i]['Player'] = passing_data[i]['Player'].str.replace('+', '', regex = False)

        rushing_data[i] = rushing_data[i].dropna()
        receiving_data[i] = receiving_data[i].dropna()
        passing_data[i] = passing_data[i].dropna()

    rushing_data[22].head()   # Check if works 


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

    categories = ['Age', 'Rush_Att', 'Rush_Yds', 'Rush_TD', 'Rush_Y/A',	'Rush_Y/G',	'Tgt',	'Rec',	'Receiving_Yds',	'Y/R',	'Receiving_TD',	'Y/Tgt',	'R/G',	'Receiving_Y/G',	'Cmp',	'Pass_Att',	'Passing_Yds',	'Pass_TD',	'Int',	'AY/A',	'Y/C',	'Pass_Y/G',	'Rate',	'Sk',	'ANY/A',	'Fumbles',	'Games_Played',	'Games_Started']

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

    #print(df[22][df[22]["Player"].str.strip() == "Derek Carr"])
    #print(df[21][df[21]["Player"].str.strip() == "Tyreek Hill"])
    #print(df[22][df[22]["Player"].str.strip() == "Josh Jacobs"])
    # all good


    # ==================================================
    # Linear Regression ==================================================
    # ==================================================
    print()
    print("========== START LIN REG ==========")
    print()

    chosen_year = 22

    # Linear Regression Test
    # Choose poistion we want to predict for 
    choice = 'QB'
    if choice == 'RB':
        X = df[chosen_year][['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Fumbles', 'Rush_Att']]
        y = df[chosen_year]['Fantasy_Points'] 
    elif choice == 'WR':
        X = df[chosen_year][['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Tgt', 'Rec', 'Fumbles']]
        y = df[chosen_year]['Fantasy_Points']
    elif choice == 'QB':
        X = df[chosen_year][['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Int', 'Sk', 'Rate']]
        y = df[chosen_year]['Fantasy_Points']

    # Split the data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean squared error:", mse)
    print("R-squared:", r2)

    # player_name = "Brian Robinson"

    # # Select the row corresponding to the player
    # player_row = df[chosen_year][df[chosen_year]['Player'].str.strip() == player_name]

    # # Extract the predictor variables for the player
    # if choice == 'QB':
    #     player_data = player_row[['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Int', 'Sk', 'Rate']]
    # elif choice == 'RB':
    #     player_data = player_row[['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Fumbles', 'Rush_Att']]
    # else:
    #     player_data = player_row[['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Tgt', 'Rec']]

    # # Make a prediction using the model
    # predicted_score = model.predict(player_data)

    # print("Lin Reg predicted score is " + predicted_score) # ==========

    # Page URL ---- RB
    url = 'https://www.fantasypros.com/nfl/projections/rb.php?week=draft'

    # Open and Pass url to Beautiful Soup
    html = urlopen(url)
    projections = BeautifulSoup(html)

    # Headers
    headers = projections.findAll('tr')[1]
    headers = [i.getText() for i in headers.findAll('th')]

    # Check out headers 
    # print(headers)

    # Get table rows into an array
    rows = projections.findAll('tr')[1:]

    # Get stats from each row
    proj = []
    for x in range(1,len(rows)):
        proj.append([col.getText() for col in rows[x].findAll('td')])

    projections_df = pd.DataFrame(proj, columns = headers[0:])

    # Keep only the player name and projections columns
    projections_df = projections_df[['Player', 'FPTS']]

    # Split the Player column that containes Name and Team into separate 'Player' and 'Tm' columns
    projections_df[['Player', 'Tm']] = projections_df['Player'].str.extract(r'^(\S+\s+\S+)\s+(.*)$')

    projections_df.drop('Tm', axis=1, inplace=True)

    # Quick Check 
    #projections_df.head()

    #print(projections_df)

    #playerlist = projections_df['Player'].values
    #print(playerlist)

    player_df = projections_df[['Player']].copy()
    #print(player_df)

    predicted_scores = []
    calc_scores = []
    enough = 0

    for playername in player_df['Player']:

        if enough == 30:
            break
        else:
            enough+=1

        calc_scores.append(df[chosen_year].loc[df[chosen_year]["Player"].str.strip() == playername, "Fantasy_Points"].values[0])

        #print(playername)
        #print(type(playername))
        #if type(playername) == float:
            #continue

        # Select the row corresponding to the player
        player_row = df[chosen_year][df[chosen_year]['Player'].str.strip() == playername]

        # Extract the predictor variables for the player
        if choice == 'QB':
            player_data = player_row[['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Int', 'Sk', 'Rate']]
        elif choice == 'RB':
            player_data = player_row[['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Fumbles', 'Rush_Att']]
        else:
            player_data = player_row[['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Tgt', 'Rec']]

        # Make a prediction using the model
        predicted_score = model.predict(player_data)
        predicted_scores.append(predicted_score[0])

        #print(playername + " " + str(predicted_score))

    parsed_projections_df = projections_df.head(30).copy()
    parsed_projections_df['FPTS'] = parsed_projections_df['FPTS'].astype(float)

    parsed_projections_df['LinReg pts'] = predicted_scores
    parsed_projections_df['true pts'] = calc_scores

    parsed_projections_df['FPTS err'] = abs(parsed_projections_df['FPTS']-parsed_projections_df['true pts']) / parsed_projections_df['true pts']
    parsed_projections_df['LinReg err'] = abs(parsed_projections_df['LinReg pts']-parsed_projections_df['true pts']) / parsed_projections_df['true pts']

    print(parsed_projections_df)

    avg_fpts_err = parsed_projections_df['FPTS err'].mean()
    avg_linreg_err = parsed_projections_df['LinReg err'].mean()

    print("fpts accuracy err = " + str(avg_fpts_err))
    print("linreg accuracy err = " + str(avg_linreg_err))

    print()
    print("========== END LIN REG ==========")
    print()
        


    # # ==================================================
    # # Ridge Regression ==================================================
    # # ==================================================
    # print("---Ridge Regression")

    # chosen_year = 21

    # # Choose position we want to predict for 
    # choice = 'RB'
    # if choice == 'RB':
    #     features = ['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Fumbles', 'Rush_Att']
    # elif choice == 'WR':
    #     features = ['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Tgt', 'Rec', 'Fumbles']
    # elif choice == 'QB':
    #     features = ['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Cmp','Int', 'Sk', 'Rate']

    # X = df[chosen_year][features]
    # y = df[chosen_year]['Fantasy_Points']

    # # Split the data into training, validation, and testing sets
    # X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)

    # # Define the hyperparameters to be tuned
    # alphas = [0.01, 0.1, 1, 10, 100]

    # # Initialize variables to keep track of best hyperparameters and performance
    # best_alpha = None
    # best_mse = float('inf')
    # best_r2 = -float('inf')

    # # Loop over hyperparameters and fit Ridge regression models
    # for alpha in alphas:
    #     # Define the Ridge regression model
    #     ridge_model = Ridge(alpha=alpha)

    #     # Train the Ridge regression model on the training set
    #     ridge_model.fit(X_train, y_train)

    #     # Generate predictions on the validation set
    #     y_val_pred = ridge_model.predict(X_val)

    #     # Calculate the mean squared error and R-squared value on the validation set
    #     val_mse = mean_squared_error(y_val, y_val_pred)
    #     val_r2 = r2_score(y_val, y_val_pred)
    #     print(alpha)
    #     print(round(val_mse, 3))
    #     print(round(val_r2,5))
    #     # Check if the current model has better performance than previous models
    #     if val_mse < best_mse:
    #         best_alpha = alpha
    #         best_mse = val_mse
    #         best_r2 = val_r2

    # # Concatenate the training and validation sets for the final model
    # X_train_val = pd.concat([X_train, X_val], axis=0)
    # y_train_val = pd.concat([y_train, y_val], axis=0)

    # # Fit the Ridge regression model with the best hyperparameters on the combined training and validation set
    # ridge_model = Ridge(alpha=best_alpha)
    # ridge_model.fit(X_train_val, y_train_val)

    # # Generate predictions on the testing set
    # y_pred = ridge_model.predict(X_test)

    # # Calculate the mean squared error and R-squared value on the testing set
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)

    # from sklearn.model_selection import GridSearchCV
    # # Choose position we want to predict for 
    # choice = 'RB'
    # if choice == 'RB':
    #     features = ['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Fumbles', 'Rush_Att']
    # elif choice == 'WR':
    #     features = ['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Tgt', 'Rec', 'Fumbles']
    # elif choice == 'QB':
    #     features = ['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Cmp','Int', 'Sk', 'Rate']

    # # Extract the relevant features and target variable
    # X = df[chosen_year][features]
    # y = df[chosen_year]['Fantasy_Points']

    # # Split the data into training, validation, and testing sets
    # X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)

    # # Define the hyperparameters to be tuned
    # param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}

    # # Define the Ridge regression model
    # ridge_model = Ridge()

    # # Define the grid search object
    # grid_search = GridSearchCV(ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error')

    # # Fit the grid search object on the training set
    # grid_search.fit(X_train, y_train)

    # # Print the best hyperparameters and performance metrics
    # print("Best hyperparameters:", grid_search.best_params_)
    # print("Best negative mean squared error:", grid_search.best_score_)
    # print("Best R-squared value:", grid_search.best_estimator_.score(X_val, y_val))

    # # Generate predictions on the testing set using the best model
    # y_pred = grid_search.predict(X_test)

    # # Calculate the mean squared error and R-squared value on the testing set
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)

    # print("Mean squared error:", round(mse, 3))
    # print("R-squared value:", round(r2,3))

    # player_name = "Jonathan Taylor"

    # # Select the row corresponding to the player
    # player_row = df[chosen_year][df[chosen_year]['Player'].str.strip() == player_name]

    # # Extract the predictor variables for the player
    # if choice == 'QB':
    #     player_data = player_row[['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Int', 'Sk', 'Rate']]
    # elif choice == 'RB':
    #     player_data = player_row[['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Fumbles', 'Rush_Att']]
    # elif choice =='WR':
    #     player_data = player_row[['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Tgt', 'Rec', 'Fumbles']]

    # # Make a prediction using the model
    # predicted_score = grid_search.predict(player_data)

    # print(predicted_score)

    # df[22][df[22]["Player"].str.strip() == player_name]



    # # Page URL ---- RB
    # url = 'https://www.fantasypros.com/nfl/projections/wr.php?week=draft'

    # # Open and Pass url to Beautiful Soup
    # html = urlopen(url)
    # projections = BeautifulSoup(html)

    # # Headers
    # headers = projections.findAll('tr')[1]
    # headers = [i.getText() for i in headers.findAll('th')]

    # # Check out headers 
    # headers

    # # Get table rows into an array
    # rows = projections.findAll('tr')[1:]

    # # Get stats from each row
    # proj = []
    # for x in range(1,len(rows)):
    #     proj.append([col.getText() for col in rows[x].findAll('td')])

    # projections_df = pd.DataFrame(proj, columns = headers[0:])

    # # Keep only the player name and projections columns
    # projections_df = projections_df[['Player', 'FPTS']]

    # # Split the Player column that containes Name and Team into separate 'Player' and 'Tm' columns
    # projections_df[['Player', 'Tm']] = projections_df['Player'].str.extract(r'^(\S+\s+\S+)\s+(.*)$')

    # projections_df.drop('Tm', axis=1, inplace=True)

    # # Quick Check 
    # projections_df.head()

    # print(projections_df)

    # """create baseline measuring system

    # """

    # # Page URL ---- RB
    # url = 'https://www.fantasypros.com/nfl/projections/rb.php?week=draft'

    # # Open and Pass url to Beautiful Soup
    # html = urlopen(url)
    # projections = BeautifulSoup(html)

    # # Headers
    # headers = projections.findAll('tr')[1]
    # headers = [i.getText() for i in headers.findAll('th')]

    # # Check out headers 
    # # print(headers)

    # # Get table rows into an array
    # rows = projections.findAll('tr')[1:]

    # # Get stats from each row
    # proj = []
    # for x in range(1,len(rows)):
    #     proj.append([col.getText() for col in rows[x].findAll('td')])

    # projections_df = pd.DataFrame(proj, columns = headers[0:])

    # # Keep only the player name and projections columns
    # projections_df = projections_df[['Player', 'FPTS']]

    # # Split the Player column that containes Name and Team into separate 'Player' and 'Tm' columns
    # projections_df[['Player', 'Tm']] = projections_df['Player'].str.extract(r'^(\S+\s+\S+)\s+(.*)$')

    # projections_df.drop('Tm', axis=1, inplace=True)

    # # Quick Check 
    # #projections_df.head()

    # #print(projections_df)

    # #playerlist = projections_df['Player'].values
    # #print(playerlist)

    # player_df = projections_df[['Player']].copy()
    # #print(player_df)

    # predicted_scores = []
    # calc_scores = []
    # enough = 0

    # for playername in player_df['Player']:

    #     if enough == 30:
    #         break
    #     else:
    #         enough+=1

    #     calc_scores.append(df[22].loc[df[22]["Player"].str.strip() == playername, "Fantasy_Points"].values[0])

    #     #print(playername)
    #     #print(type(playername))
    #     #if type(playername) == float:
    #         #continue

    #     # Select the row corresponding to the player
    #     player_row = df[22][df[22]['Player'].str.strip() == playername]

    #     # Extract the predictor variables for the player
    #     if choice == 'QB':
    #         player_data = player_row[['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Int', 'Sk', 'Rate']]
    #     elif choice == 'RB':
    #         player_data = player_row[['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Fumbles', 'Rush_Att']]
    #     elif choice =='WR':
    #         player_data = player_row[['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Tgt', 'Rec', 'Fumbles']]
    
    # # Make a prediction using the model
    # predicted_score = grid_search.predict(player_data)
    # predicted_scores.append(predicted_score[0])

    # #print(playername + " " + str(predicted_score))

    # parsed_projections_df = projections_df.head(30).copy()
    # parsed_projections_df['FPTS'] = parsed_projections_df['FPTS'].astype(float)

    # parsed_projections_df['RidgeReg pts'] = predicted_scores
    # parsed_projections_df['true pts'] = calc_scores

    # parsed_projections_df['FPTS err'] = abs(parsed_projections_df['FPTS']-parsed_projections_df['true pts']) / parsed_projections_df['true pts']
    # parsed_projections_df['RidgeReg err'] = abs(parsed_projections_df['RidgeReg pts']-parsed_projections_df['true pts']) / parsed_projections_df['true pts']

    # pd.set_option('display.max_rows',500)
    # pd.set_option('display.max_columns',504)
    # pd.set_option('display.width',1000)

    # print(parsed_projections_df)

    # avg_fpts_err = parsed_projections_df['FPTS err'].mean()
    # avg_ridgereg_err = parsed_projections_df['RidgeReg err'].mean()

    # print("fpts accuracy err = " + str(avg_fpts_err))
    # print("ridgereg accuracy err = " + str(avg_ridgereg_err))

    # """now qb

    # """

    # # Page URL ---- RB
    # url = 'https://www.fantasypros.com/nfl/projections/rb.php?week=draft'

    # # Open and Pass url to Beautiful Soup
    # html = urlopen(url)
    # projections = BeautifulSoup(html)

    # # Headers
    # headers = projections.findAll('tr')[1]
    # headers = [i.getText() for i in headers.findAll('th')]

    # # Check out headers 
    # # print(headers)

    # # Get table rows into an array
    # rows = projections.findAll('tr')[1:]

    # # Get stats from each row
    # proj = []
    # for x in range(1,len(rows)):
    #     proj.append([col.getText() for col in rows[x].findAll('td')])

    # projections_df = pd.DataFrame(proj, columns = headers[0:])

    # # Keep only the player name and projections columns
    # projections_df = projections_df[['Player', 'FPTS']]

    # # Split the Player column that containes Name and Team into separate 'Player' and 'Tm' columns
    # projections_df[['Player', 'Tm']] = projections_df['Player'].str.extract(r'^(\S+\s+\S+)\s+(.*)$')

    # projections_df.drop('Tm', axis=1, inplace=True)

    # # Quick Check 
    # #projections_df.head()

    # #print(projections_df)

    # #playerlist = projections_df['Player'].values
    # #print(playerlist)

    # player_df = projections_df[['Player']].copy()
    # #print(player_df)

    # predicted_scores = []
    # calc_scores = []
    # enough = 0

    # for playername in player_df['Player']:

    #     if enough == 30:
    #         break
    #     else:
    #         enough+=1

    #     calc_scores.append(df[22].loc[df[22]["Player"].str.strip() == playername, "Fantasy_Points"].values[0])

    #     #print(playername)
    #     #print(type(playername))
    #     #if type(playername) == float:
    #         #continue

    #     # Select the row corresponding to the player
    #     player_row = df[22][df[22]['Player'].str.strip() == playername]

    #     # Extract the predictor variables for the player
    #     if choice == 'QB':
    #         player_data = player_row[['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Int', 'Sk', 'Rate']]
    #     elif choice == 'RB':
    #         player_data = player_row[['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Fumbles', 'Rush_Att']]
    #     elif choice =='WR':
    #         player_data = player_row[['Passing_Yds', 'Rush_Yds', 'Receiving_Yds', 'Pass_TD', 'Rush_TD', 'Receiving_TD', 'Age', 'Tgt', 'Rec', 'Fumbles']]
    
    # # Make a prediction using the model
    # predicted_score = grid_search.predict(player_data)
    # predicted_scores.append(predicted_score[0])

    # #print(playername + " " + str(predicted_score))

    # parsed_projections_df = projections_df.head(30).copy()
    # parsed_projections_df['FPTS'] = parsed_projections_df['FPTS'].astype(float)

    # parsed_projections_df['RidgeReg pts'] = predicted_scores
    # parsed_projections_df['true pts'] = calc_scores

    # parsed_projections_df['FPTS err'] = abs(parsed_projections_df['FPTS']-parsed_projections_df['true pts']) / parsed_projections_df['true pts']
    # parsed_projections_df['RidgeReg err'] = abs(parsed_projections_df['RidgeReg pts']-parsed_projections_df['true pts']) / parsed_projections_df['true pts']

    # pd.set_option('display.max_rows',500)
    # pd.set_option('display.max_columns',504)
    # pd.set_option('display.width',1000)

    # print(parsed_projections_df)

    # avg_fpts_err = parsed_projections_df['FPTS err'].mean()
    # avg_ridgereg_err = parsed_projections_df['RidgeReg err'].mean()

    # print("fpts accuracy err = " + str(avg_fpts_err))
    # print("ridgereg accuracy err = " + str(avg_ridgereg_err))



if __name__ == '__main__':
    main()