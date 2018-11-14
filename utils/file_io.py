"""
General purpose functions for file i/o.
"""
import csv

def write_list_to_file(fname, l, mode='a+'):
    with open(fname, mode) as f:
        for line in l:
            f.write(line + "\n")

def write_rows_to_csv(fname, l):
    with open(fname, 'w+') as f:
        writer = csv.writer(f)
        writer.writerows(l)

def write_row_to_csv(fname,r,mode='a+'):
    with open(fname,mode) as f:
        writer = csv.writer(f)
        writer.writerow(r)