import argparse
import sys
import csv
import os
import time

# Import scraping modules
from urllib.request import urlopen
from bs4 import BeautifulSoup


# parse data from pro-football-reference with BeautifulSoup4
def parse_stats(year):

    passing_url = 'https://www.pro-football-reference.com/years/' + year +'/passing.htm'
    rushing_url = 'https://www.pro-football-reference.com/years/' + year +'/rushing.htm'
    receiving_url = 'https://www.pro-football-reference.com/years/' + year +'/receiving.htm'

    # Set sleep 5 seconds since error 429 occured, expect 15s * 23 = ~5min to run
    wait = 5
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

# write data into a csv file
def writeCSV(data, filename):

    folder = "csv"
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, filename)

    with open(filepath, 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(data)


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
        writeCSV(passing_stats[i], 'pa200' + str(i) + 'raw.csv')
        writeCSV(rushing_stats[i], 'ru200' + str(i) + 'raw.csv')
        writeCSV(receiving_stats[i], 're200' + str(i) + 'raw.csv')
        passing_headers.append(pa_headers)
        rushing_headers.append(ru_headers)
        receiving_headers.append(re_headers)
        writeCSV(passing_headers[i], 'pa200' + str(i) + 'head.csv')
        writeCSV(rushing_headers[i], 'ru200' + str(i) + 'head.csv')
        writeCSV(receiving_headers[i], 're200' + str(i) + 'head.csv')
    for i in range(10, 23):
        pa_stats, ru_stats, re_stats, pa_headers, ru_headers, re_headers = parse_stats("20" + str(i))
        passing_stats.append(pa_stats)
        rushing_stats.append(ru_stats)
        receiving_stats.append(re_stats)
        writeCSV(passing_stats[i], 'pa20' + str(i) + 'raw.csv')
        writeCSV(rushing_stats[i], 'ru20' + str(i) + 'raw.csv')
        writeCSV(receiving_stats[i], 're20' + str(i) + 'raw.csv')
        passing_headers.append(pa_headers)
        rushing_headers.append(ru_headers)
        receiving_headers.append(re_headers)
        writeCSV(passing_headers[i], 'pa20' + str(i) + 'head.csv')
        writeCSV(rushing_headers[i], 'ru20' + str(i) + 'head.csv')
        writeCSV(receiving_headers[i], 're20' + str(i) + 'head.csv')


if __name__ == '__main__':
    main()