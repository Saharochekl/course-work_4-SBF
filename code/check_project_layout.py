#!/usr/bin/env python3
"""Check code, notebook syntax and article inputs; never execute notebook cells.

From code/: py check_project_layout.py --with-products
The optional product check reads metadata only, not FITS pixel arrays.
"""
import argparse
import ast
import json
from pathlib import Path

from IPython.core.inputtransformer2 import TransformerManager

from publish_article_assets import publish_article_assets
from sbf_paths import PROJECT_ROOT, load_project_json


def check_sources():
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
    if isinstance(value, dict):
        for item in value.values():
            yield from local_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from local_paths(item)
    elif isinstance(value, str) and value.startswith(str(PROJECT_ROOT) + '/'):
        yield Path(value)


def check_products():
    root = PROJECT_ROOT / 'runs'
    groups = {
        'F150 source': sorted((root / 'sbf2_go3055/batch').glob('NGC_*_result.json')),
        'F150 normalized': sorted((root / 'sbf2_normalized_winsor/batch/results').glob('NGC_*_result.json')),
        'F090 published': sorted((root / 'sbf_f090w_go3055/products').glob('NGC_*/products.json')),
    }
    issues, refs = [], set()
    for label, paths in groups.items():
        if len(paths) != 14:
            issues.append(f'{label}: expected 14 records, found {len(paths)}')
        for path in paths:
            payload = load_project_json(path)
            refs.update(local_paths(payload))
            if label == 'F090 published':
                for key in ['source_result', 'final_result']:
                    refs.update(local_paths(load_project_json(payload[key])))
        print(f'{label}: {len(paths)}/14 metadata records')
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
