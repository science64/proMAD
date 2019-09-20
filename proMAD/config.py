from pathlib import Path

app_name = 'proMAD'
version_number = (0, 3, 1)
version = f'{version_number[0]}.{version_number[1]}.{version_number[2]}'
app_author = 'Anna Jaeschke; Hagen Eckert'
url = 'https://proMAD.dev'

base_dir = Path(__file__).absolute().parent
array_data_folder = base_dir / 'data' / 'array'
template_folder = base_dir / 'data' / 'templates'
allowed_load_version = (0, 3, 0)

scale = 30
