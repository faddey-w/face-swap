import nbformat as nbf


def main():
    code_files = ["reface/dataloader.py"]
    cells = []

    for filename in code_files:
        cells.append(nbf.v4.new_markdown_cell(f"### {filename}"))

        code_text = open(filename).read()
        lines = code_text.splitlines()
        for i, line in enumerate(lines):
            if "import " in line and "reface" in line:
                lines[i] = "# " + line

        cells.append(nbf.v4.new_code_cell("\n".join(lines)))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    fname = "compiled.ipynb"

    with open(fname, "w") as f:
        nbf.write(nb, f)


if __name__ == "__main__":
    main()
