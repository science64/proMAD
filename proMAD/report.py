import io
import os
import warnings
from datetime import datetime
from tempfile import NamedTemporaryFile

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.drawing.image import Image as opImage
from openpyxl.styles import NamedStyle, Alignment, Font
from openpyxl.utils import get_column_letter

from . import config


def report_excel(aa, file, norm='hist_raw', name=None, additional_info=[]):
    """
    Export results in an Excel file.

    Parameters
    ----------
    aa:
        ArrayAnalyse instant
    norm: str
            evaluation strategy selection (see ArrayAnalyse.evaluate)
    file:
        file can be a path to a file (a string), a file-like object, or a path-like object;
    name: str
        membrane name
    additional_info: list
        a list with pairs of additional information ('name', 'value')
    """

    if not aa.is_finalized:
        warnings.warn('Data collection needs to be finalized to generate a report.', RuntimeWarning)
        return None
    wb = Workbook()

    highlight = NamedStyle(name="highlight")
    highlight.font = Font(bold=True, size=20)
    wb.add_named_style(highlight)

    bold = NamedStyle(name="bold")
    bold.font = Font(bold=True)
    wb.add_named_style(bold)

    bold_right = NamedStyle(name="bold_right")
    bold_right.font = Font(bold=True)
    bold_right.alignment = Alignment(horizontal='right')
    wb.add_named_style(bold_right)

    bold_center = NamedStyle(name="bold_center")
    bold_center.font = Font(bold=True)
    bold_center.alignment = Alignment(horizontal='center')
    wb.add_named_style(bold_center)

    # Overview worksheet
    ws = wb.active
    if name is None:
        ws['A1'] = 'Membrane Overview'
    else:
        ws['A1'] = f'{name} Overview'
    ws['A1'].style = highlight
    ws.title = "Overview"
    data = aa.evaluate(norm=norm)
    row_offset = 4
    column_offset = 2
    for entry in data:
        if isinstance(entry['value'], float):
            ws.cell(column=entry['position'][1] + column_offset, row=entry['position'][0] + row_offset,
                    value=entry['value'])
        else:
            ws.cell(column=entry['position'][1] + column_offset, row=entry['position'][0] + row_offset,
                    value=entry['value'][0])

    for column in range(sum(aa.array_data['net_layout_x'])):
        ws.cell(column=column + column_offset, row=row_offset - 1, value=column + 1).style = bold_center

    for row in range(sum(aa.array_data['net_layout_y'])):
        ws.cell(column=column_offset - 1, row=row + row_offset, value=get_column_letter(row + 1)).style = bold_right

    alignment_img_raw = io.BytesIO()
    aa.figure_alignment(file=alignment_img_raw)
    alignment_img = opImage(alignment_img_raw)
    ws[f"B{sum(aa.array_data['net_layout_y']) + row_offset + 1}"] = 'Fig. 1: Overview and alignment check '
    ws.add_image(alignment_img, f"B{sum(aa.array_data['net_layout_y'])+row_offset+2}")

    # Result worksheet
    ws = wb.create_sheet()
    ws.title = "Results"
    new_rows = []
    data = aa.evaluate(norm=norm, double_spot=True)
    for entry in data:
        if isinstance(entry['value'], float):
            new_rows.append([entry['info'][0], str(entry['position']), entry['value']])
        else:
            new_rows.append([entry['info'][0], str(entry['position']), entry['value'][0]])

    if name is None:
        ws['A1'] = 'Membrane Results'
    else:
        ws['A1'] = f'{name} Results'
    ws['A1'].style = highlight
    ws.append([])
    ws.append(('Name', 'Position', 'Value'))
    ws['A3'].style = bold
    ws['B3'].style = bold
    ws['C3'].style = bold_right

    data_start = 4
    for row in sorted(new_rows, key=lambda s: s[2], reverse=True):
        ws.append(row)
    ws.column_dimensions['A'].width = 20
    ws.column_dimensions['C'].width = 12

    cutoff = max(min(len(new_rows), 15), int(len(new_rows)*0.2))
    values = Reference(ws, min_col=3, min_row=data_start, max_col=3, max_row=cutoff+data_start)
    categories = Reference(ws, min_col=1, min_row=data_start, max_col=1, max_row=cutoff+data_start)
    chart = BarChart()
    chart.add_data(values)
    chart.set_categories(categories)
    chart.y_axis.title = 'Concentration [mol/spot]'
    chart.title = f'Top spots'
    chart.width = 30
    chart.height = 15
    chart.gapWidth = 25
    chart.legend = None
    ws.add_chart(chart, "E3")

    # Technical worksheet
    ws = wb.create_sheet()
    ws.title = "Info"

    ws['A1'] = f'Technical data'
    ws['A1'].style = highlight
    now = datetime.now()
    tech_data_list = [
        ('Date', now.date()),
        ('Time', now.time()),
        ('', ''),
        ('Program', 'proMAD'),
        ('Version', config.version),
        ('URL', config.url),
    ]
    tech_data_list += additional_info
    ws.column_dimensions['B'].width = 20
    for n, content in enumerate(tech_data_list):
        ws.cell(column=1, row=n+3, value=content[0]).style = bold_right
        ws.cell(column=2, row=n+3, value=content[1])

    if isinstance(file, os.PathLike) or isinstance(file, str):
        wb.save(file)
    elif isinstance(file, (io.RawIOBase, io.BufferedIOBase)):
        with NamedTemporaryFile() as tmp:
            wb.save(tmp.name)
            tmp.seek(0)
            file.write(tmp.read())
