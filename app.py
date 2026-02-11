import os
import time
import streamlit as st
from utilities.preprocessing import train_model, classify_pdf
from utilities.invoice_extraction import extract_invoice_data, save_invoice_to_excel

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PDF Document Classifier", layout="centered")
st.title("AI Document Classification")
st.write("Upload a PDF to classify it")

# -----------------------------
# Folder to save uploaded PDFs
# -----------------------------
UPLOAD_FOLDER = r"C://Users/divij/OneDrive/Documents/invoice"
DESTINATION_FOLDER = r"C://Users/divij/OneDrive/Documents/Python Scripts/PDFClassification/uploaded_pdfs"
os.makedirs(DESTINATION_FOLDER, exist_ok=True)

# -----------------------------
# Function to generate dynamic labels from file names
# -----------------------------
def get_pdf_labels(pdf_folder):
    """
    Automatically assign labels to PDFs based on file name patterns.
    Example:
        - Files containing 'invoice' → label 'Invoice'
        - Files containing 'bill' → label 'Bill of Lading'
    """
    pdf_labels = {}
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            lower_name = file.lower()
            if "invoice" in lower_name:
                pdf_labels[file] = "Invoice"
            elif "bill" in lower_name:
                pdf_labels[file] = "Bill of Lading"
            else:
                pdf_labels[file] = "Unknown"
    return pdf_labels

# -----------------------------
# Train model once and cache
# -----------------------------
@st.cache_resource
def load_model():
    pdf_labels = get_pdf_labels(UPLOAD_FOLDER)
    return train_model(UPLOAD_FOLDER, pdf_labels)

with st.spinner("Loading model..."):
    model, embed_model, label_encoder = load_model()

# -----------------------------
# PDF Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    # -------------------------
    # Save uploaded file
    # -------------------------
    timestamp = int(time.time())
    save_name = f"{timestamp}_{uploaded_file.name}"
    save_path = os.path.join(DESTINATION_FOLDER, save_name)
    # save_address = save_path+save_name

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File saved as: {save_name}")

    # -------------------------
    # Classify uploaded PDF
    # -------------------------
    with st.spinner("Classifying document..."):
        try:
            prediction = classify_pdf(uploaded_file, model, embed_model, label_encoder)
            st.success("Classification complete")
            st.subheader("Document Type")
            st.markdown(f"### **{prediction}**")
            if prediction == "Invoice":
                df = extract_invoice_data(uploaded_file)
                save_invoice_to_excel(df, "invoice_data.xlsx")
                st.success("Invoice data extracted and saved to Excel")
                # st.dataframe(df)

        except Exception as e:
            st.error(f"Could not classify PDF: {e}")

    
