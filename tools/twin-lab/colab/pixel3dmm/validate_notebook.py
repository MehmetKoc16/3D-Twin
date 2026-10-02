"""Validate the head notebook with nbformat and py_compile; no Colab execution."""

from pathlib import Path
import py_compile
import shutil
import uuid
import nbformat

HERE = Path(__file__).resolve().parent


def main() -> None:
    notebook = nbformat.read(HERE.parent / "pixel3dmm_head.ipynb", as_version=4)
    nbformat.validate(notebook)
    count = 0
    directory = HERE / ".cache" / ("extracted-cells-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    try:
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                if cell.execution_count is not None or cell.outputs:
                    raise ValueError("Notebook must have no executions or saved outputs")
                path = Path(directory) / f"cell_{index}.py"
                path.write_text(cell.source, encoding="utf-8")
                py_compile.compile(str(path), doraise=True)
                count += 1
        for path in HERE.glob("*.py"):
            py_compile.compile(str(path), cfile=str(Path(directory) / (path.stem + ".pyc")), doraise=True)
    finally:
        shutil.rmtree(directory)
    print(f"Valid nbformat 4; {count} code cells and all helpers passed py_compile. No Colab execution.")


if __name__ == "__main__":
    main()
