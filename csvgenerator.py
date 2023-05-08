import argparse
import sys

import csv
import os

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

def parse_stats(year):

    # Page URL
    passing_url = 'https://www.pro-football-reference.com/years/' + year +'/passing.htm'
    rushing_url = 'https://www.pro-football-reference.com/years/' + year +'/rushing.htm'
    receiving_url = 'https://www.pro-football-reference.com/years/' + year +'/receiving.htm'

    # Open and Pass url to Beautiful Soup
    html = urlopen(passing_url)
    passing_stats = BeautifulSoup(html)
    time.sleep(5)
    html = urlopen(rushing_url)
    rushing_stats = BeautifulSoup(html)
    time.sleep(5)
    html = urlopen(receiving_url)
    receiving_stats = BeautifulSoup(html)
    time.sleep(5)

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

    passing_stats = []
    passing_headers = []
    rushing_stats = []
    rushing_headers = []
    receiving_stats = []
    receiving_headers = []

    for i in range(10):
        pa_stats, ru_stats, re_stats, pa_headers, ru_headers, re_headers = parse_stats("200" + str(i))
        passing_stats.append(pa_stats)
        rushing_stats.append(ru_stats)
        receiving_stats.append(re_stats)
        passing_headers.append(pa_headers)
        rushing_headers.append(ru_headers)
        receiving_headers.append(re_headers)
        writeCSV(passing_stats[i], 'pa200' + str(i) + 'raw.csv')
        writeCSV(rushing_stats[i], 'ru200' + str(i) + 'raw.csv')
        writeCSV(receiving_stats[i], 're200' + str(i) + 'raw.csv')
    for i in range(10, 23):
        pa_stats, ru_stats, re_stats, pa_headers, ru_headers, re_headers = parse_stats("20" + str(i))
        passing_stats.append(pa_stats)
        rushing_stats.append(ru_stats)
        receiving_stats.append(re_stats)
        passing_headers.append(pa_headers)
        rushing_headers.append(ru_headers)
        receiving_headers.append(re_headers)
        writeCSV(passing_stats[i], 'pa20' + str(i) + 'raw.csv')
        writeCSV(rushing_stats[i], 'ru20' + str(i) + 'raw.csv')
        writeCSV(receiving_stats[i], 're20' + str(i) + 'raw.csv')



if __name__ == '__main__':
    main()