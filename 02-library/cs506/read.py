import csv

def read_csv(csv_file_path):
    """
        Given a path to a csv file, return a matrix (list of lists)
        in row major.
    """     
    return [
        [int(entry) if entry.isnumeric() else entry for entry in row]
        for row in csv.reader(open(csv_file_path, 'r'))
    ]

