# Streamlit application

import streamlit as st
from app import user_input, setting_chromaDB, get_pdf_into_chunks, GoogleGenerativeAIEmbeddings, get_pdf
def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chat with PDF using Gemini 💁")

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf(pdf_docs)
                    text_chunks = get_pdf_into_chunks(raw_text)
                    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    setting_chromaDB(text_chunks, embedding)
                    st.success("PDF processing completed and embeddings stored!")

    # Input for user questions
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        with st.spinner("Searching for an answer..."):
            response = user_input(user_question)  # Call user_input function to get the response
            st.markdown(f"### Answer: {response}")  # Display the response

# Run the app
if __name__ == "__main__":
    main()