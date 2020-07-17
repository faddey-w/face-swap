import nbformat as nbf
import textwrap
import yaml


def main():
    code_files = [
        "reface/env.py",
        "reface/config.py",
        "reface/utils.py",
        "reface/data_lib.py",
        "reface/face_recognizer.py",
        "reface/colab.py",
        "play.py",
    ]
    cells = [
        nbf.v4.new_code_cell(

        ),
        nbf.v4.new_code_cell(
            textwrap.dedent(
                """
                import numpy as np
                import cv2
                from matplotlib import pyplot as plt
                from PIL import Image
                %matplotlib inline
                """
            )
        )
    ]

    for filename in code_files:
        code_text = open(filename).read()
        lines = code_text.splitlines()
        for i, line in enumerate(lines):
            if "import " in line and "reface" in line:
                lines[i] = "# " + line
        lines.insert(0, f"#@title {filename}")

        cells.append(nbf.v4.new_code_cell("\n".join(lines)))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    fname = "compiled.ipynb"

    with open(fname, "w") as f:
        nbf.write(nb, f)


if __name__ == "__main__":
    main()
