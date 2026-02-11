import pdfplumber
import pandas as pd
import re


def extract_invoice_data(pdf_file):
    pdf_file.seek(0)
    text = ""
    tables = []

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            tables.extend(page.extract_tables())

    invoice_no = extract_pattern(text, r"invoice\s*#?\s*([A-Z0-9\-]+)")
    invoice_date = extract_pattern(text, r"date[:\s]*([\d./-]+)")
    country_origin = extract_pattern(text, r"country\s*of\s*origin[:\-]?\s*(\w+)")

    df = extract_line_items(tables)

    if not df.empty:
        df["Invoice No"] = invoice_no
        df["Invoice Date"] = invoice_date
        df["Country of Origin"] = country_origin

    return df


def extract_line_items(tables):
    for table in tables:
        if not table or len(table) < 2:
            continue

        header = [str(col).lower() for col in table[0]]

        if any("description" in col for col in header):
            return pd.DataFrame(table[1:], columns=table[0])

    return pd.DataFrame()


def extract_pattern(text, pattern):
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else None


def save_invoice_to_excel(df, output_path):
    df.to_excel(output_path, index=False)


