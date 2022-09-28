def write_file(file_path: str, content: str):
    with open(file_path, 'w') as file:
        return file.write(content)


def load_file(file_path: str) -> str:
    with open(file_path) as file:
        return file.read()


def write_csv_row(file_path, row):
    import csv
    with open(file_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(row)
