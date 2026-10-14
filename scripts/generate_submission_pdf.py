"""
scripts/generate_submission_pdf.py
==================================
Compiles outputs/final_submission/manuscript.md and figures into a publication-grade
submission PDF: outputs/final_submission/Trustworthy_Multi_Fruit_Quality_Grading.pdf.
Uses headless Chrome/Edge to produce an IEEE-style formatted paper with tables and figures.
"""

import os
import sys
import base64
import subprocess
import re
import shutil
import mistune

def get_base64_image(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def format_latex_math(latex_str):
    """Converts LaTeX math strings into clean, beautiful HTML typography with Unicode symbols."""
    s = latex_str.strip()
    
    # Common macro replacements
    s = s.replace(r"\mathbb{R}", "&#8477;")
    s = s.replace(r"\mathbf{x}", "<b>x</b>")
    s = s.replace(r"\mathbf{w}", "<b>w</b>")
    s = s.replace(r"\Delta", "&Delta;")
    s = s.replace(r"\sigma", "&sigma;")
    s = s.replace(r"\approx", "&asymp;")
    s = s.replace(r"\times", "&times;")
    s = s.replace(r"\le", "&le;")
    s = s.replace(r"\ge", "&ge;")
    s = s.replace(r"\pm", "&plusmn;")
    s = s.replace(r"\in", "&isin;")
    s = s.replace(r"\mid", "|")
    s = s.replace(r"\dots", "...")
    s = s.replace(r"\ln", "ln")
    s = s.replace(r"\mathbb{I}", "<b>1</b>")
    s = s.replace(r"\theta_0", "&theta;<sub>0</sub>")
    s = s.replace(r"\delta_j", "&delta;<sub>j</sub>")
    s = s.replace(r"\delta", "&delta;")
    s = s.replace(r"\hat{y}", "<i>y&#770;</i>")
    s = s.replace(r"\beta", "&beta;")
    s = s.replace(r"\sum", "&sum;")
    s = s.replace(r"\alpha", "&alpha;")

    # Handle \text{...}
    s = re.sub(r'\\text\{([^}]+)\}', r'\1', s)

    # Handle fractions \frac{num}{den}
    s = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1) / (\2)', s)

    # Handle superscripts and subscripts with braces: ^{...} and _{...}
    s = re.sub(r'\^\{([^}]+)\}', r'<sup>\1</sup>', s)
    s = re.sub(r'\_\{([^}]+)\}', r'<sub>\1</sub>', s)

    # Handle single character superscripts and subscripts: ^x and _x
    s = re.sub(r'\^([0-9a-zA-Z])', r'<sup>\1</sup>', s)
    s = re.sub(r'\_([0-9a-zA-Z])', r'<sub>\1</sub>', s)

    return s

def main():
    print("Generating Publication-Grade Submission PDF...")
    root_dir = os.path.abspath(".")
    manuscript_path = os.path.join(root_dir, "outputs", "final_submission", "manuscript.md")
    figures_dir = os.path.join(root_dir, "outputs", "final_submission", "figures")
    out_pdf_dir = os.path.join(root_dir, "outputs", "final_submission")
    out_pdf_path = os.path.join(out_pdf_dir, "Trustworthy_Multi_Fruit_Quality_Grading.pdf")
    lock_pdf_path = os.path.join(root_dir, "outputs", "phase57_final_submission_lock", "Trustworthy_Multi_Fruit_Quality_Grading.pdf")

    with open(manuscript_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Pre-encode images as base64
    fig_b64 = {}
    for i in range(1, 7):
        fig_name = f"figure{i}"
        for fname in os.listdir(figures_dir):
            if fname.startswith(fig_name) and fname.endswith(".png"):
                fig_path = os.path.join(figures_dir, fname)
                fig_b64[i] = (get_base64_image(fig_path), fname)
                break

    # Save math before Markdown parsing using pure alphanumeric tokens
    equations = []
    def save_block_eq(match):
        idx = len(equations)
        equations.append((True, match.group(1).strip()))
        return f"XYZBLOCKMATH{idx}XYZ"

    def save_inline_eq(match):
        idx = len(equations)
        equations.append((False, match.group(1).strip()))
        return f"XYZINLINEMATH{idx}XYZ"

    # Display math ($$...$$)
    processed_md = re.sub(r'\$\$(.*?)\$\$', save_block_eq, md_text, flags=re.DOTALL)
    # Inline math ($...$)
    processed_md = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', save_inline_eq, processed_md)

    markdown = mistune.create_markdown(plugins=['table', 'strikethrough'])
    html_body = markdown(processed_md)

    # Restore math with formatted HTML
    for idx, (is_block, eq_content) in enumerate(equations):
        clean_eq = format_latex_math(eq_content)
        if is_block:
            repl = f'<div class="display-equation"><span class="equation-content">{clean_eq}</span></div>'
            html_body = html_body.replace(f"XYZBLOCKMATH{idx}XYZ", repl)
        else:
            repl = f'<span class="inline-math">{clean_eq}</span>'
            html_body = html_body.replace(f"XYZINLINEMATH{idx}XYZ", repl)

    # Check that all math was replaced
    assert "XYZBLOCKMATH" not in html_body, "Unreplaced display equation token found!"
    assert "XYZINLINEMATH" not in html_body, "Unreplaced inline equation token found!"

    # Insert figures with captions at their thematic locations
    fig_html = {
        1: f'''<div class="figure-container">
            <img src="{fig_b64[1][0]}" alt="Figure 1: Dataset Composition">
            <div class="figure-caption"><b>Fig. 1.</b> Multi-Fruit Dataset Composition across Five Species and Four Acquisition Domains (Studio <code>fruits360</code>, Market Bins <code>real_quality</code>, Packing Conveyors <code>real_grade</code>, and Controlled Lab <code>real_severity</code>; total N=56,036).</div>
        </div>''',
        2: f'''<div class="figure-container">
            <img src="{fig_b64[2][0]}" alt="Figure 2: QWK Distribution Across 30 Seeds">
            <div class="figure-caption"><b>Fig. 2.</b> Empirical QWK Distribution across 30 Deterministic Random Seeds for Config A (Image Only) versus Auxiliary Conditioning under Historical Split V2 and 100% Group Session Holdout.</div>
        </div>''',
        3: f'''<div class="figure-container">
            <img src="{fig_b64[3][0]}" alt="Figure 3: Paired Statistical Contrasts">
            <div class="figure-caption"><b>Fig. 3.</b> Paired Difference Distributions (&Delta;QWK) and 95% Confidence Intervals for B-A, C-A, and D-A Contrasts across 30 Paired Seeds, Demonstrating Consistent Negative Effects of Categorical Conditioning.</div>
        </div>''',
        4: f'''<div class="figure-container">
            <img src="{fig_b64[4][0]}" alt="Figure 4: Real-Image OOD Domain Inversion">
            <div class="figure-caption"><b>Fig. 4.</b> Domain-Stratified Real-Image OOD Performance Inversion on 5,000 Unencountered Species Images: Mahalanobis Distance Collapses on Market Bins (18.08% AUROC) while Free Energy Score Remains Resilient (92.49% AUROC).</div>
        </div>''',
        5: f'''<div class="figure-container">
            <img src="{fig_b64[5][0]}" alt="Figure 5: Selective Prediction Risk-Coverage Curves">
            <div class="figure-caption"><b>Fig. 5.</b> Selective Prediction Risk-Coverage Profile and Reliability Diagrams: Rejection at 90% Coverage Cuts Ordinal MAE by 84.70% (Seed 83) and 81.05% (30-Seed Aggregate), Outperforming Random Rejection.</div>
        </div>''',
        6: f'''<div class="figure-container">
            <img src="{fig_b64[6][0]}" alt="Figure 6: Inference Throughput and Latency Benchmarks">
            <div class="figure-caption"><b>Fig. 6.</b> Edge Inference Latency and Throughput Benchmarks across Batch Sizes (1 to 32) and Numerical Precisions (FP16 vs FP32) on an NVIDIA RTX 3050 Laptop GPU.</div>
        </div>'''
    }

    # Inject figures near respective sections
    if "<h2>3.1 Multi-Fruit Dataset Composition</h2>" in html_body:
        html_body = html_body.replace("<h2>3.1 Multi-Fruit Dataset Composition</h2>", "<h2>3.1 Multi-Fruit Dataset Composition</h2>\n" + fig_html[1])
    elif "<h3>Table 1:" in html_body:
        html_body = html_body.replace("<h3>Table 1:", fig_html[1] + "\n<h3>Table 1:")

    if "<h3>Table 2:" in html_body:
        html_body = html_body.replace("<h3>Table 2:", fig_html[2] + "\n<h3>Table 2:")

    if "<h2>5.2 Real-Image Open-Set Recognition Benchmark</h2>" in html_body:
        html_body = html_body.replace("<h2>5.2 Real-Image Open-Set Recognition Benchmark</h2>", "<h2>5.2 Real-Image Open-Set Recognition Benchmark</h2>\n" + fig_html[3])

    if "<h3>Table 4:" in html_body:
        html_body = html_body.replace("<h3>Table 4:", fig_html[4] + "\n<h3>Table 4:")

    if "<h2>5.3 Calibration and Selective Prediction</h2>" in html_body:
        html_body = html_body.replace("<h2>5.3 Calibration and Selective Prediction</h2>", "<h2>5.3 Calibration and Selective Prediction</h2>\n" + fig_html[5])

    if "<h2>5.4 Computational Efficiency and Parameter Census</h2>" in html_body:
        html_body = html_body.replace("<h2>5.4 Computational Efficiency and Parameter Census</h2>", "<h2>5.4 Computational Efficiency and Parameter Census</h2>\n" + fig_html[6])

    # Beautiful IEEE Transactions / ACM style publication CSS
    html_document = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trustworthy Open-Set Ordinal Multi-Fruit Quality Grading with Monotonic Ordinal Regression and Real-Image OOD Evaluation</title>
<style>
@page {{
    size: letter;
    margin: 18mm 16mm 18mm 16mm;
    @bottom-center {{
        content: counter(page);
        font-family: "Times New Roman", Times, serif;
        font-size: 9pt;
    }}
}}
body {{
    font-family: "Times New Roman", Times, "Liberation Serif", serif;
    font-size: 10pt;
    line-height: 1.35;
    color: #111111;
    margin: 0;
    padding: 0;
    text-align: justify;
    background-color: #ffffff;
}}
h1:first-of-type {{
    font-size: 18pt;
    font-weight: bold;
    text-align: center;
    line-height: 1.25;
    margin-top: 0;
    margin-bottom: 12pt;
    color: #0b1f3a;
}}
.metadata {{
    text-align: center;
    font-size: 9.5pt;
    margin-bottom: 14pt;
    line-height: 1.4;
    color: #333333;
}}
.abstract-box {{
    margin: 12pt 20pt 16pt 20pt;
    padding: 10pt 14pt;
    background-color: #f8fafc;
    border-left: 3pt solid #1e3a8a;
    border-radius: 2pt;
    font-size: 9.2pt;
    line-height: 1.35;
}}
.abstract-title {{
    font-weight: bold;
    font-size: 9.5pt;
    text-transform: uppercase;
    color: #1e3a8a;
    letter-spacing: 0.5px;
    margin-bottom: 4pt;
}}
.keywords {{
    margin-top: 6pt;
    font-size: 8.8pt;
    font-style: italic;
    color: #4b5563;
}}
h1 {{
    font-size: 12pt;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #0f172a;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 3pt;
    margin-top: 16pt;
    margin-bottom: 6pt;
    page-break-after: avoid;
}}
h2 {{
    font-size: 10.5pt;
    font-weight: bold;
    color: #1e293b;
    margin-top: 12pt;
    margin-bottom: 4pt;
    page-break-after: avoid;
}}
h3 {{
    font-size: 9.8pt;
    font-weight: bold;
    font-style: italic;
    color: #334155;
    margin-top: 10pt;
    margin-bottom: 3pt;
    page-break-after: avoid;
}}
p {{
    margin-top: 0;
    margin-bottom: 6pt;
    text-indent: 12pt;
}}
ul, ol {{
    margin-top: 2pt;
    margin-bottom: 6pt;
    padding-left: 20pt;
}}
li {{
    margin-bottom: 2pt;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10pt 0 12pt 0;
    font-size: 8pt;
    line-height: 1.25;
    page-break-inside: avoid;
}}
table, th, td {{
    border: none;
}}
th {{
    border-top: 1.5pt solid #0f172a;
    border-bottom: 1pt solid #0f172a;
    background-color: #f1f5f9;
    padding: 4pt 3pt;
    font-weight: bold;
    text-align: left;
    color: #0f172a;
}}
td {{
    padding: 3.5pt 3pt;
    border-bottom: 0.5pt solid #e2e8f0;
    text-align: left;
}}
tr:last-child td {{
    border-bottom: 1.5pt solid #0f172a;
}}
blockquote {{
    margin: 8pt 10pt;
    padding: 6pt 10pt;
    background-color: #f8fafc;
    border-left: 2.5pt solid #64748b;
    font-size: 8.8pt;
    font-style: italic;
    page-break-inside: avoid;
}}
.figure-container {{
    margin: 12pt auto;
    text-align: center;
    page-break-inside: avoid;
    max-width: 96%;
}}
.figure-container img {{
    max-width: 92%;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 3pt;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.figure-caption {{
    font-size: 8.5pt;
    line-height: 1.3;
    color: #334155;
    margin-top: 4pt;
    text-align: justify;
    padding: 0 10pt;
}}
.display-equation {{
    margin: 6pt 0;
    text-align: center;
    font-family: "Cambria Math", "Times New Roman", serif;
    font-size: 9.8pt;
    page-break-inside: avoid;
}}
.inline-math {{
    font-family: "Cambria Math", "Times New Roman", serif;
    font-style: italic;
}}
code {{
    font-family: "Courier New", Courier, monospace;
    font-size: 8.5pt;
    background-color: #f1f5f9;
    padding: 1pt 3pt;
    border-radius: 2pt;
}}
hr {{
    border: none;
    border-top: 0.5pt solid #cbd5e1;
    margin: 14pt 0;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>'''

    temp_html = os.path.join(root_dir, "scratch", "submission_manuscript.html")
    os.makedirs(os.path.join(root_dir, "scratch"), exist_ok=True)
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_document)
    print(f"Generated clean HTML layout at {temp_html} ({os.path.getsize(temp_html):,d} bytes)")

    # Locate Chrome or Edge
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    browser = chrome_path if os.path.exists(chrome_path) else edge_path
    print(f"Using rendering engine: {browser}")

    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={out_pdf_path}",
        temp_html
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"ERROR: Browser failed to generate PDF. Return code: {res.returncode}")
        print(res.stderr)
        return False

    if not os.path.exists(out_pdf_path) or os.path.getsize(out_pdf_path) == 0:
        print("ERROR: Generated PDF is empty or missing.")
        return False

    pdf_size = os.path.getsize(out_pdf_path)
    print(f"Successfully generated PDF at {out_pdf_path} ({pdf_size:,d} bytes)")

    # Copy to lock directory as well
    shutil.copy2(out_pdf_path, lock_pdf_path)
    print(f"Mirrored submission PDF to {lock_pdf_path}")

    # Inspect PDF with fitz (PyMuPDF)
    import fitz
    doc = fitz.open(out_pdf_path)
    page_count = len(doc)
    print(f"Inspected PDF: Total Pages = {page_count}, Title = {doc.metadata.get('title', 'None')}")
    doc.close()

    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
