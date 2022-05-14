import numpy as np
import csv

def reader_csv(csvfilename):
    matrix = np.empty((0, 10), int)
    with open(csvfilename, newline="") as csvfile:
        lines = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in lines:
            new_row = np.matrix(row[0])
            matrix = np.append(matrix, new_row, axis=0)
    return np.array(matrix)