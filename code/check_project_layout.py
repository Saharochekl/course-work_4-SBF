#!/usr/bin/env python3
"""Check code, notebook syntax and article inputs; never execute notebook cells.

From code/: py check_project_layout.py --with-products
The optional product check reads metadata only, not FITS pixel arrays.
"""
import argparse
import ast
import csv
import json
from pathlib import Path

from IPython.core.inputtransformer2 import TransformerManager

from publish_article_assets import publish_article_assets
from sbf_paths import PROJECT_ROOT, load_project_json


def check_sources():
    """RU: компиляция без выполнения. EN: compile source without executing it."""
    code = PROJECT_ROOT / 'code'
    scripts = sorted(code.glob('*.py'))
    for path in scripts:
        ast.parse(path.read_text(), filename=str(path))
    notebooks = sorted(code.glob('*.ipynb'))
    transform = TransformerManager()
    count = 0
    for path in notebooks:
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = transform.transform_cell(''.join(cell['source']))
                compile(source, f'{path.name}:cell{index}', 'exec',
                        flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                count += 1
    print(f'Syntax: {len(scripts)} Python files; {len(notebooks)} notebooks, {count} cells (not executed).')


def local_paths(value):
    """RU: пути текущего проекта в JSON. EN: yield in-checkout metadata paths."""
    if isinstance(value, dict):
        for item in value.values():
            yield from local_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from local_paths(item)
    elif isinstance(value, str) and value.startswith(str(PROJECT_ROOT) + '/'):
        yield Path(value)


def check_products():
    """RU: состав выборки и ссылки, не новый фит. EN: membership and links, not a fit."""
    root = PROJECT_ROOT / 'runs'
    # RU: список целей задаёт манифест, не независимая магическая константа 14.
    # EN: the manifest is the single source of target membership.
    with (PROJECT_ROOT / 'code/targets_go3055_manifest.csv').open(newline='') as handle:
        expected = {row['target'] for row in csv.DictReader(handle)}
    groups = {
        'F150 source': sorted((root / 'sbf2_go3055/batch').glob('NGC_*_result.json')),
        'F150 normalized': sorted((root / 'sbf2_normalized_winsor/batch/results').glob('NGC_*_result.json')),
        'F090 published': sorted((root / 'sbf_f090w_go3055/products').glob('NGC_*/products.json')),
    }
    issues, refs = [], set()
    for label, paths in groups.items():
        found = []
        for path in paths:
            payload = load_project_json(path)
            found.append(payload['galaxy'])
            if label != 'F090 published' and payload.get('status') != 'ok':
                issues.append(f'{label}: unsuccessful result {path.name}')
            refs.update(local_paths(payload))
            if label == 'F090 published':
                for key in ['source_result', 'final_result']:
                    linked = load_project_json(payload[key])
                    if linked.get('status') != 'ok' or linked['galaxy'] != payload['galaxy']:
                        issues.append(f'{label}: inconsistent {key} for {payload["galaxy"]}')
                    refs.update(local_paths(linked))
        if set(found) != expected or len(found) != len(expected):
            issues.append(f'{label}: wrong membership; missing={sorted(expected - set(found))}, '
                          f'extra={sorted(set(found) - expected)}, records={len(found)}')
        print(f'{label}: {len(paths)}/{len(expected)} metadata records')
    issues.extend(f'Missing product: {p.relative_to(PROJECT_ROOT)}' for p in sorted(refs) if not p.exists())
    if issues:
        raise FileNotFoundError('\n'.join(issues))
    print(f'Products: {len(refs)} referenced paths exist (pixel arrays not read).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-products', action='store_true')
    args = parser.parse_args()
    check_sources()
    print(f'Article figures: {publish_article_assets(check=True)} copies checked.')
    if args.with_products:
        check_products()
