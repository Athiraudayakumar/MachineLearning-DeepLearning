import os
import re
import pdfplumber
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# PDF utilities
# -----------------------------
def extract_pdf_content(pdf_file):
    """
    Extract text and tables from a PDF file.
    pdf_file: str path or BytesIO object
    """
    text = ""
    tables = []

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            tables.extend(page.extract_tables())

    return text, tables


def flatten_tables(tables):
    """Flatten PDF tables into a single string."""
    return " ".join(
        " ".join(str(cell) for cell in row if cell)
        for table in tables
        for row in table
    )


def preprocess_text(text):
    """Clean text: lowercase + remove special characters."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text) # Removes extra spaces, tabs, newlines with spaces
    text = re.sub(r"[^a-z0-9 ]", " ", text) # keeps only lower case, digits, space
    return text.strip()


# -----------------------------
# Load & train model
# -----------------------------
def train_model(pdf_folder, pdf_labels):
    """
    Train XGBoost classifier on PDF documents.
    pdf_folder: folder where labeled PDFs are stored
    pdf_labels: dict {file_name: label}
    """
    documents = []
    labels = []

    for file_name, label in pdf_labels.items():
        pdf_path = os.path.join(pdf_folder, file_name)
        print(pdf_path)
        if not os.path.exists(pdf_path):
            continue  # Skip missing files
        text, tables = extract_pdf_content(pdf_path)
        table_text = flatten_tables(tables)
        doc_text = preprocess_text(text + " " + table_text)

        documents.append(doc_text)
        labels.append(label)

    if not documents:
        raise ValueError("No PDF files found for training!")

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Sentence embeddings
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    X = embed_model.encode(documents)

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        eval_metric="mlogloss",
        use_label_encoder=False
    )
    model.fit(X, y)

    return model, embed_model, label_encoder


# -----------------------------
# Prediction
# -----------------------------
def classify_pdf(pdf_file, model, embed_model, label_encoder, threshold=0.60):
    pdf_file.seek(0)

    text, tables = extract_pdf_content(pdf_file)
    table_text = flatten_tables(tables)
    doc_text = preprocess_text(text + " " + table_text)
    print(doc_text)

    X_new = embed_model.encode([doc_text])


    # Predict probabilities
    probs = model.predict_proba(X_new)[0]
    max_prob = probs.max()
    pred_id = probs.argmax()

    # DEBUG (important for interview)
    print("Class probabilities:", probs)
    print("Max probability:", max_prob)

    # Confidence-based rejection
    if max_prob < threshold: # minimm threshold required
        return "Unknown"

    return label_encoder.inverse_transform([pred_id])[0]


