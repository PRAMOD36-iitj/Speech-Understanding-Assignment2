import sys
import subprocess

try:
    import pypdf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
    import pypdf

reader = pypdf.PdfReader("d:/OFFICE-WORKS/pramod_behara/assignment2/files (1)/DOC-20260406-WA0010_pdf.pdf")
for page in reader.pages:
    print(page.extract_text())
