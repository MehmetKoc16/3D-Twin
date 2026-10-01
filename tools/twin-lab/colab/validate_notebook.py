"""Validate nbformat and py_compile every extracted code cell; do not run Colab."""

from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import nbformat

HERE = Path(__file__).resolve().parent


def main() -> None:
    notebook = nbformat.read(HERE / "trellis2_shape.ipynb", as_version=4)
    nbformat.validate(notebook)
    count = 0
    paths = []
    directory = HERE / ".cache" / ("extracted-cells-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    try:
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                path = Path(directory) / f"cell_{index}.py"
                path.write_text(cell.source, encoding="utf-8")
                paths.append(str(path))
                count += 1
        subprocess.run([sys.executable, "-m", "py_compile", *paths], check=True)
    finally:
        shutil.rmtree(directory)
    print(f"Valid nbformat 4 notebook; {count} extracted code cells passed py_compile. No inference executed.")


if __name__ == "__main__":
    main()
