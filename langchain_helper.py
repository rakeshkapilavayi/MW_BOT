from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import pickle
from dotenv import load_dotenv

load_dotenv()
vectordb_file_path = "faiss_index"
pkl_file_path = "fais_db.pkl"

def create_vector_db():
    try:
        loader = CSVLoader(file_path="codebasics_faqs.csv", source_column="Prompt", encoding="utf-8")
        data = loader.load()
    except FileNotFoundError:
        print("Error: The file 'codebasics_faqs.csv' was not found.")
        return False
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False

    try:
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    except Exception as e:
        print(f"Error initializing Instructor embeddings: {e}. Falling back to all-MiniLM-L6-v2.")
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Error initializing fallback embeddings: {e}")
            return False

    try:
        vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    except Exception as e:
        print(f"Error creating FAISS vector database: {e}")
        return False

    try:
        vectordb.save_local(vectordb_file_path)
        print(f"FAISS vector database saved locally to '{vectordb_file_path}'.")
    except Exception as e:
        print(f"Error saving FAISS index locally: {e}")
        return False

    try:
        with open(pkl_file_path, "wb") as f:
            pickle.dump(vectordb, f)
        print(f"FAISS vector database saved as '{pkl_file_path}'.")
    except Exception as e:
        print(f"Error saving FAISS vector database as .pkl: {e}")
        return False

    return True

def get_qa_chain():
    try:
        try:
            embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
        except Exception as e:
            print(f"Error initializing Instructor embeddings: {e}. Falling back to all-MiniLM-L6-v2.")
            try:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            except Exception as e:
                print(f"Error initializing fallback embeddings: {e}")
                return None

        try:
            with open(pkl_file_path, "rb") as f:
                vectordb = pickle.load(f)
            print(f"FAISS vector database loaded from '{pkl_file_path}'.")
        except FileNotFoundError:
            print(f"Error: The file '{pkl_file_path}' was not found. Loading from local index.")
            try:
                vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
                print(f"FAISS vector database loaded from '{vectordb_file_path}'.")
            except Exception as e:
                print(f"Error loading FAISS index from local folder: {e}")
                return None

        retriever = vectordb.as_retriever(score_threshold=0.7)

        prompt_template = """Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
        If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}"""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ.get("GOOGLE_API_KEY"), temperature=0.1)

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        return chain
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        return None