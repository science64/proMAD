from pathlib import Path

app_name = 'proMAD'
version_number = (0, 1, 0)
version = f'{version_number[0]}.{version_number[1]}.{version_number[2]}'
app_author = 'Anna Jaeschke; Hagen Eckert'
base_dir = Path(__file__).absolute().parent
array_data_folder = base_dir / 'array_data'
allowed_load_version = (0, 1, 0)

scale = 30
