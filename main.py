import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("MONEY WALA Q&A 🌱")

if st.button("Create Knowledgebase"):
    with st.spinner("Creating knowledge base..."):
        success = create_vector_db()
        if success:
            st.success("Knowledge base created successfully!")
        else:
            st.error("Failed to create knowledge base. Check the console for details.")

question = st.text_input("Ask a question about Money Wala:", placeholder="e.g., Where do I update my password??")

if question:
    chain = get_qa_chain()
    if chain is None:
        st.error("Failed to load QA chain. Check the console for details.")
    else:
        with st.spinner("Generating answer..."):
            try:
                response = chain({"query": question})
                st.header("Answer")
                st.write(response["result"])
            except Exception as e:
                st.error(f"Error generating answer: {e}")