# This file helps in converting the entire project folder contents (only relevant) to a single pdf 
# Import all required libraries 
import os
from pathlib import Path
import nbformat
from fpdf import FPDF

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PDF = BASE_DIR / "Tesla_Stock_Project_All_In_One.pdf"

README = BASE_DIR / "README.md"
APP = BASE_DIR / "app.py"
NB = BASE_DIR / "tesla_stock_prediction.ipynb"

# ---------- PDF Helper ----------
class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, "Tesla Stock Price Prediction Project", ln=True, align="C")
        self.ln(2)
        self.set_draw_color(180, 180, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

def add_title(pdf: FPDF, title: str):
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 8, title)
    pdf.ln(2)

def add_text_block(pdf: FPDF, text: str, font_size=10):
    pdf.set_font("Courier", "", font_size)  # monospace safe for code/text
    # FPDF needs latin-1 by default; replace unsupported chars safely
    safe = text.replace("\t", "    ")
    safe = safe.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 5, safe)
    pdf.ln(2)

def add_file(pdf: FPDF, path: Path, title: str):
    add_title(pdf, title)
    if not path.exists():
        add_text_block(pdf, f"[Missing file] {path.name}")
        return
    content = path.read_text(encoding="utf-8", errors="replace")
    add_text_block(pdf, content)

def add_tree(pdf: FPDF):
    add_title(pdf, "Project Tree Structure")
    lines = []
    for root, dirs, files in os.walk(BASE_DIR):
        # ignore noisy folders
        dirs[:] = [d for d in dirs if d not in {".ipynb_checkpoints", ".git", "__pycache__", ".venv", "venv"}]
        rel = Path(root).relative_to(BASE_DIR)
        indent = "  " * len(rel.parts)
        lines.append(f"{indent}{rel.as_posix()}/" if rel.parts else f"{BASE_DIR.name}/")
        for f in sorted(files):
            if f.endswith((".pyc",)):
                continue
            lines.append(f"{indent}  {f}")
    add_text_block(pdf, "\n".join(lines))

def add_notebook(pdf: FPDF, nb_path: Path):
    add_title(pdf, "Notebook: tesla_stock_prediction.ipynb")
    if not nb_path.exists():
        add_text_block(pdf, "[Missing notebook] tesla_stock_prediction.ipynb")
        return

    nb = nbformat.read(nb_path, as_version=4)

    for i, cell in enumerate(nb.cells, start=1):
        if cell.cell_type == "markdown":
            add_title(pdf, f"Markdown Cell {i}")
            add_text_block(pdf, cell.source)
        elif cell.cell_type == "code":
            add_title(pdf, f"Code Cell {i}")
            add_text_block(pdf, cell.source)

            # outputs (text only)
            outputs_text = []
            for out in cell.get("outputs", []):
                if out.get("output_type") == "stream" and "text" in out:
                    outputs_text.append(out["text"])
                elif out.get("output_type") == "execute_result":
                    data = out.get("data", {})
                    if "text/plain" in data:
                        outputs_text.append(str(data["text/plain"]))
                elif out.get("output_type") == "error":
                    outputs_text.append("\n".join(out.get("traceback", [])))

            if outputs_text:
                add_title(pdf, f"Outputs (Cell {i})")
                add_text_block(pdf, "\n".join(outputs_text))

def main():
    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    # 1) Tree
    add_tree(pdf)

    pdf.add_page()
    # 2) README
    add_file(pdf, README, "README.md")

    pdf.add_page()
    # 3) app.py
    add_file(pdf, APP, "app.py")

    pdf.add_page()
    # 4) Notebook
    add_notebook(pdf, NB)

    pdf.output(str(OUTPUT_PDF))
    print(f"\n✅ PDF created successfully: {OUTPUT_PDF}\n")

if __name__ == "__main__":
    main()
