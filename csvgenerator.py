import argparse
import sys

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

#Import file writing modeles
import csv
import os

def parse_stats(year):

    # Page URL
    passing_url = 'https://www.pro-football-reference.com/years/' + year +'/passing.htm'
    rushing_url = 'https://www.pro-football-reference.com/years/' + year +'/rushing.htm'
    receiving_url = 'https://www.pro-football-reference.com/years/' + year +'/receiving.htm'

    # Open and Pass url to Beautiful Soup
    html = urlopen(passing_url)
    passing_stats = BeautifulSoup(html)
    html = urlopen(rushing_url)
    rushing_stats = BeautifulSoup(html)
    html = urlopen(receiving_url)
    receiving_stats = BeautifulSoup(html)

    # Headers
    passing_col_headers = passing_stats.findAll('tr')[0]
    passing_col_headers = [i.getText() for i in passing_col_headers.findAll('th')]

    rushing_col_headers = rushing_stats.findAll('tr')[1]
    rushing_col_headers = [i.getText() for i in rushing_col_headers.findAll('th')]

    receiving_col_headers = receiving_stats.findAll('tr')[0]
    receiving_col_headers = [i.getText() for i in receiving_col_headers.findAll('th')]

    return passing_stats, rushing_stats, receiving_stats, passing_col_headers, rushing_col_headers, receiving_col_headers


def writeCSV(headers, filename):

    folder = "csv"
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, filename)

    with open(filepath, 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)


def main():

    parse_stats("2021")
    passing_2022_stats, rushing_2022_stats, receiving_2022_stats, passing_headers_2022, rushing_headers_2022, receiving_headers_2022 = parse_stats("2022")
    writeCSV(passing_2022_stats, 'pa2022raw.csv')
    writeCSV(rushing_2022_stats, 'ru2022raw.csv')
    writeCSV(receiving_2022_stats, 're2022raw.csv')

    passing_2021_stats, rushing_2021_stats, receiving_2021_stats, passing_headers_2021, rushing_headers_2021, receiving_headers_2021 = parse_stats("2021")
    writeCSV(passing_2021_stats, 'pa2021raw.csv')
    writeCSV(rushing_2021_stats, 'ru2021raw.csv')
    writeCSV(receiving_2021_stats, 're2021raw.csv')

    passing_2020_stats, rushing_2020_stats, receiving_2020_stats, passing_headers_2020, rushing_headers_2020, receiving_headers_2020 = parse_stats("2020")
    writeCSV(passing_2020_stats, 'pa2020raw.csv')
    writeCSV(rushing_2020_stats, 'ru2020raw.csv')
    writeCSV(receiving_2020_stats, 're2020raw.csv')


if __name__ == '__main__':
    main()