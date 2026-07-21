# -*- coding: utf-8 -*-
"""Tabular Peak Analysis export without numerical fitting dependencies."""

from __future__ import annotations

import csv
import math
import os
import re
import zipfile
from xml.etree import ElementTree as ET


PEAK_EXPORT_COLUMNS = (
    'File',
    'Peak',
    'tau / s',
    'f / Hz',
    'Peak height / ohm',
    'Resistance contribution / ohm',
    'Contribution / %',
    'FWHM / decades',
    'Component index',
)

_MAIN_NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
_REL_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
_PACKAGE_REL_NS = 'http://schemas.openxmlformats.org/package/2006/relationships'


def peak_export_rows(display_name, entry):
    """Return all serializable metrics attached to one analyzed spectrum."""
    rows = []
    for index, result in enumerate(list(getattr(entry, 'peak_results', []) or [])):
        rows.append((
            str(display_name),
            str(result.get('name', f'Peak {index + 1}')),
            _finite_number(result.get('tau_s')),
            _finite_number(result.get('frequency_hz')),
            _finite_number(result.get('height_ohm')),
            _finite_number(result.get('resistance_ohm')),
            _finite_number(result.get('fraction_percent')),
            _finite_number(result.get('fwhm_decades')),
            int(result.get('component_index', index)),
        ))
    return rows


def write_peak_csv(path, display_name, entry):
    """Write one file's Peak Analysis results as UTF-8 CSV."""
    path = _ensure_extension(path, '.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(PEAK_EXPORT_COLUMNS)
        writer.writerows(peak_export_rows(display_name, entry))
    return path


def write_peak_workbook(path, datasets):
    """Write multiple files to one dependency-free XLSX workbook.

    ``datasets`` is an iterable of ``(display_name, entry)`` pairs. Each pair
    becomes one worksheet, matching the order shown in the Files panel.
    """
    path = _ensure_extension(path, '.xlsx')
    sheets = []
    used_names = set()
    for display_name, entry in datasets:
        sheet_name = _unique_sheet_name(display_name, used_names)
        rows = [PEAK_EXPORT_COLUMNS]
        rows.extend(peak_export_rows(display_name, entry))
        sheets.append((sheet_name, rows))
    if not sheets:
        raise ValueError('No Peak Analysis results are available for export.')

    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', _content_types_xml(len(sheets)))
        archive.writestr('_rels/.rels', _package_relationships_xml())
        archive.writestr('xl/workbook.xml', _workbook_xml([name for name, _ in sheets]))
        archive.writestr('xl/_rels/workbook.xml.rels', _workbook_relationships_xml(len(sheets)))
        archive.writestr('xl/styles.xml', _styles_xml())
        for index, (_, rows) in enumerate(sheets, start=1):
            archive.writestr(f'xl/worksheets/sheet{index}.xml', _worksheet_xml(rows))
    return path


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ''
    return number if math.isfinite(number) else ''


def _ensure_extension(path, extension):
    path = str(path or '').strip()
    if not path:
        raise ValueError('Empty export path.')
    if not path.lower().endswith(extension):
        path += extension
    return path


def _unique_sheet_name(display_name, used_names):
    base = re.sub(r'[\\/*?:\[\]]', '_', str(display_name or 'Peaks')).strip(" '")
    base = (base or 'Peaks')[:31]
    candidate = base
    suffix = 2
    while candidate.casefold() in used_names:
        tail = f' ({suffix})'
        candidate = f'{base[:31 - len(tail)]}{tail}'
        suffix += 1
    used_names.add(candidate.casefold())
    return candidate


def _column_name(index):
    result = ''
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _xml_bytes(root):
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)


def _worksheet_xml(rows):
    ET.register_namespace('', _MAIN_NS)
    worksheet = ET.Element(f'{{{_MAIN_NS}}}worksheet')
    views = ET.SubElement(worksheet, f'{{{_MAIN_NS}}}sheetViews')
    view = ET.SubElement(views, f'{{{_MAIN_NS}}}sheetView', {'workbookViewId': '0'})
    ET.SubElement(view, f'{{{_MAIN_NS}}}pane', {'ySplit': '1', 'topLeftCell': 'A2', 'state': 'frozen'})
    columns = ET.SubElement(worksheet, f'{{{_MAIN_NS}}}cols')
    for index, width in enumerate((22, 18, 16, 16, 21, 31, 18, 20, 18), start=1):
        ET.SubElement(columns, f'{{{_MAIN_NS}}}col', {
            'min': str(index), 'max': str(index), 'width': str(width), 'customWidth': '1'
        })
    sheet_data = ET.SubElement(worksheet, f'{{{_MAIN_NS}}}sheetData')
    for row_index, values in enumerate(rows, start=1):
        row = ET.SubElement(sheet_data, f'{{{_MAIN_NS}}}row', {'r': str(row_index)})
        for column_index, value in enumerate(values, start=1):
            reference = f'{_column_name(column_index)}{row_index}'
            attributes = {'r': reference}
            if row_index == 1:
                attributes['s'] = '1'
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cell = ET.SubElement(row, f'{{{_MAIN_NS}}}c', attributes)
                ET.SubElement(cell, f'{{{_MAIN_NS}}}v').text = str(value)
            else:
                attributes['t'] = 'inlineStr'
                cell = ET.SubElement(row, f'{{{_MAIN_NS}}}c', attributes)
                inline = ET.SubElement(cell, f'{{{_MAIN_NS}}}is')
                ET.SubElement(inline, f'{{{_MAIN_NS}}}t').text = str(value)
    if rows:
        last_cell = f'{_column_name(len(rows[0]))}{len(rows)}'
        ET.SubElement(worksheet, f'{{{_MAIN_NS}}}autoFilter', {'ref': f'A1:{last_cell}'})
    return _xml_bytes(worksheet)


def _workbook_xml(sheet_names):
    ET.register_namespace('', _MAIN_NS)
    ET.register_namespace('r', _REL_NS)
    workbook = ET.Element(f'{{{_MAIN_NS}}}workbook')
    sheets = ET.SubElement(workbook, f'{{{_MAIN_NS}}}sheets')
    for index, name in enumerate(sheet_names, start=1):
        ET.SubElement(sheets, f'{{{_MAIN_NS}}}sheet', {
            'name': name,
            'sheetId': str(index),
            f'{{{_REL_NS}}}id': f'rId{index}',
        })
    return _xml_bytes(workbook)


def _workbook_relationships_xml(sheet_count):
    relationships = ET.Element('Relationships', {'xmlns': _PACKAGE_REL_NS})
    for index in range(1, sheet_count + 1):
        ET.SubElement(relationships, 'Relationship', {
            'Id': f'rId{index}',
            'Type': f'{_REL_NS}/worksheet',
            'Target': f'worksheets/sheet{index}.xml',
        })
    ET.SubElement(relationships, 'Relationship', {
        'Id': f'rId{sheet_count + 1}',
        'Type': f'{_REL_NS}/styles',
        'Target': 'styles.xml',
    })
    return _xml_bytes(relationships)


def _package_relationships_xml():
    relationships = ET.Element('Relationships', {'xmlns': _PACKAGE_REL_NS})
    ET.SubElement(relationships, 'Relationship', {
        'Id': 'rId1',
        'Type': f'{_REL_NS}/officeDocument',
        'Target': 'xl/workbook.xml',
    })
    return _xml_bytes(relationships)


def _content_types_xml(sheet_count):
    types = ET.Element('Types', {
        'xmlns': 'http://schemas.openxmlformats.org/package/2006/content-types'
    })
    ET.SubElement(types, 'Default', {
        'Extension': 'rels',
        'ContentType': 'application/vnd.openxmlformats-package.relationships+xml',
    })
    ET.SubElement(types, 'Default', {'Extension': 'xml', 'ContentType': 'application/xml'})
    ET.SubElement(types, 'Override', {
        'PartName': '/xl/workbook.xml',
        'ContentType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml',
    })
    ET.SubElement(types, 'Override', {
        'PartName': '/xl/styles.xml',
        'ContentType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml',
    })
    for index in range(1, sheet_count + 1):
        ET.SubElement(types, 'Override', {
            'PartName': f'/xl/worksheets/sheet{index}.xml',
            'ContentType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml',
        })
    return _xml_bytes(types)


def _styles_xml():
    ET.register_namespace('', _MAIN_NS)
    style = ET.Element(f'{{{_MAIN_NS}}}styleSheet')
    fonts = ET.SubElement(style, f'{{{_MAIN_NS}}}fonts', {'count': '2'})
    ET.SubElement(fonts, f'{{{_MAIN_NS}}}font')
    bold_font = ET.SubElement(fonts, f'{{{_MAIN_NS}}}font')
    ET.SubElement(bold_font, f'{{{_MAIN_NS}}}b')
    fills = ET.SubElement(style, f'{{{_MAIN_NS}}}fills', {'count': '2'})
    ET.SubElement(ET.SubElement(fills, f'{{{_MAIN_NS}}}fill'), f'{{{_MAIN_NS}}}patternFill', {'patternType': 'none'})
    ET.SubElement(ET.SubElement(fills, f'{{{_MAIN_NS}}}fill'), f'{{{_MAIN_NS}}}patternFill', {'patternType': 'gray125'})
    borders = ET.SubElement(style, f'{{{_MAIN_NS}}}borders', {'count': '1'})
    ET.SubElement(borders, f'{{{_MAIN_NS}}}border')
    style_xfs = ET.SubElement(style, f'{{{_MAIN_NS}}}cellStyleXfs', {'count': '1'})
    ET.SubElement(style_xfs, f'{{{_MAIN_NS}}}xf', {'numFmtId': '0', 'fontId': '0', 'fillId': '0', 'borderId': '0'})
    cell_xfs = ET.SubElement(style, f'{{{_MAIN_NS}}}cellXfs', {'count': '2'})
    ET.SubElement(cell_xfs, f'{{{_MAIN_NS}}}xf', {'numFmtId': '0', 'fontId': '0', 'fillId': '0', 'borderId': '0', 'xfId': '0'})
    ET.SubElement(cell_xfs, f'{{{_MAIN_NS}}}xf', {'numFmtId': '0', 'fontId': '1', 'fillId': '0', 'borderId': '0', 'xfId': '0', 'applyFont': '1'})
    cell_styles = ET.SubElement(style, f'{{{_MAIN_NS}}}cellStyles', {'count': '1'})
    ET.SubElement(cell_styles, f'{{{_MAIN_NS}}}cellStyle', {'name': 'Normal', 'xfId': '0', 'builtinId': '0'})
    return _xml_bytes(style)
