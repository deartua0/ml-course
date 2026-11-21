"""Run the original basic-layer tests against modules.py."""
import ast
import json
import unittest
from pathlib import Path

import numpy as np
import torch

import modules


def main():
    torch.set_num_threads(1)
    path = Path(__file__).with_name("test_modules.ipynb")
    notebook = json.loads(path.read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "class TestLayers(" in "".join(cell["source"])
    )
    test_class = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "TestLayers"
    )
    tree = ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[]))
    namespace = dict(vars(modules), np=np, numpy=np, torch=torch, unittest=unittest)
    exec(compile(tree, str(path), "exec"), namespace)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(namespace["TestLayers"])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
