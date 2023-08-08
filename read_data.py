import pandas as pd
import re
import os


def group_lines_by_patterns(patterns, files, folder):
    data1 = []
    data2 = []
    for file in files:
        file_path = os.path.join(folder, file)
        with open(file_path, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for pattern in patterns:
                    match = re.match(pattern, line)
                    if match:
                        key = match.group(1)
                        value = match.group(2)
                        if pattern == patterns[0]:
                            data1.append((key, value))
                        elif pattern == patterns[1]:
                            data2.append((key, value))
                        break
    return data1, data2


def read_data(pattern1, pattern2):
    folder = os.path.join(os.path.dirname(__file__), '..', 'data')
    files = os.listdir(folder)

    # Group the lines based on the patterns
    patterns = [pattern1, pattern2]
    data1, data2 = group_lines_by_patterns(patterns, files, folder)

    # Convert the grouped data to DataFrames
    df1 = pd.DataFrame(data1, columns=['Key', 'Value'])
    df2 = pd.DataFrame(data2, columns=['Key', 'Value'])

    return df1, df2
