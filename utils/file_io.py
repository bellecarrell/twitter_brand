"""
General purpose functions for file i/o.
"""

def write_list_to_file(fname, el, mode='a+'):
    with open(fname, mode) as f:
        for line in el:
            f.write(line + "\n")