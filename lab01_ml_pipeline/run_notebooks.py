"""Execute the submission notebooks and update their readable Markdown copies."""
import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--check-reproducibility', action='store_true',
                    help='Compare text results and PNG plots against the saved run.')
args = parser.parse_args()
root = Path(__file__).resolve().parent


def output_signature(notebook):
    """Ignore execution timestamps and HTML IDs, retain actual results and plots."""
    signature = []
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        outputs = []
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                # Jupyter может разбить один и тот же текст на разные сообщения.
                # Сравниваем сам текст, а не границы этих сообщений.
                if outputs and isinstance(outputs[-1], tuple):
                    last_name, last_text = outputs[-1]
                    if last_name == output.name:
                        outputs[-1] = (last_name, last_text + output.text)
                    else:
                        outputs.append((output.name, output.text))
                else:
                    outputs.append((output.name, output.text))
            elif output.output_type in ('display_data', 'execute_result'):
                saved_data = {}
                for key in ('text/plain', 'image/png'):
                    if key in output.data:
                        saved_data[key] = output.data[key]
                outputs.append(saved_data)
            elif output.output_type == 'error':
                raise RuntimeError(f"Saved notebook contains an error: {output.ename}")
        signature.append(outputs)
    return signature


for path in sorted(root.glob('lab01_part*.ipynb')):
    print(f'Executing {path.name}', flush=True)
    notebook = nbformat.read(path, as_version=4)
    previous = output_signature(notebook)
    NotebookClient(notebook, timeout=1200, kernel_name='python3',
                   resources={'metadata': {'path': str(root)}}).execute()
    nbformat.validate(notebook)
    actual = output_signature(notebook)
    if args.check_reproducibility and actual != previous:
        raise AssertionError(f'Results changed in {path.name}; original saved run retained.')
    nbformat.write(notebook, path)

    # Source only: the submission ipynb retains all numeric outputs and plots.
    blocks = []
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            blocks.append(f'```python\n{cell.source.rstrip()}\n```')
        else:
            blocks.append(cell.source.rstrip())
    path.with_suffix('.md').write_text('\n\n'.join(blocks) + '\n', encoding='utf-8')
    images = 0
    for cell in notebook.cells:
        for output in cell.get('outputs', []):
            if 'image/png' in output.get('data', {}):
                images += 1
    print(f'OK: {len(notebook.cells)} cells, {images} figures', flush=True)
