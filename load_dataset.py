import pdfplumber
import os

pdf_folder = "circulars_pdfs"
texts = []

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        path = os.path.join(pdf_folder, file)
        with pdfplumber.open(path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
            if len(text.strip()) == 0:
                print(f"[WARNING] {file} may be scanned/image PDF — no text found")
                texts.append({"filename": file, "text": ""})
            else:
                print(f"{file} is text-based, first 200 chars:\n{text[:200]}\n")
                texts.append({"filename": file, "text": text.strip()})

import pandas as pd

df = pd.DataFrame(texts)
df.to_csv("circulars.csv", index=False)
print("CSV saved with", len(df), "records")
