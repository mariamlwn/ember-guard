"""Dump every table + page text from a PDF (e.g. NASA/TM-20210011385) for review.
Usage: python -I extract_tables.py path/to/report.pdf out_dir [first_page last_page]"""
import sys, os, csv, pdfplumber
pdf, out = sys.argv[1], sys.argv[2]
a = int(sys.argv[3]) if len(sys.argv) > 3 else 1
os.makedirs(out, exist_ok=True)
with pdfplumber.open(pdf) as doc:
    b = int(sys.argv[4]) if len(sys.argv) > 4 else len(doc.pages)
    with open(os.path.join(out, "text.txt"), "w") as t:
        for i in range(a - 1, b):
            p = doc.pages[i]
            t.write(f"\n===== PDF page {i+1} =====\n{p.extract_text() or ''}\n")
            for j, tab in enumerate(p.extract_tables()):
                with open(os.path.join(out, f"p{i+1:03d}_t{j}.csv"), "w", newline="") as f:
                    csv.writer(f).writerows(tab)
print("done", out)
