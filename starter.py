import streamlit as st
import os
import json
import tempfile
from pathlib import Path
import pandas as pd
import docx2txt
import pymupdf4llm
from PIL import Image
import pytesseract
import mimetypes
import re
import traceback
import tiktoken
from altair.vegalite.v4.api import Chart

# LangChain / Google Generative AI / etc.
from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document

from dotenv import load_dotenv
load_dotenv(override=True)

pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"

# สร้าง LLM
def create_chat_model():
    return AzureChatOpenAI(
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version="2024-08-01-preview",
        temperature=0.5,
        max_tokens=1000
    )

# สร้างและอัปเดต Vector Store
def setup_vectorstore(documents):
    if not documents:
        st.warning("No valid text extracted from the uploaded or cached files.")
        return None
    try:
        embeddings = AzureOpenAIEmbeddings(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment="bbik-embedding-small", 
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version="2024-08-01-preview",
        )
        vector_store = FAISS.from_documents(documents, embeddings)
        print("[DEBUG] Vector store setup completed with new docs.")
        return vector_store
    except Exception as e:
        st.error(f"Error setting up vector store: {e}")
        traceback.print_exc()
        return None

def add_documents_to_vectorstore(vector_store, documents):
    try:
        if not documents or not vector_store:
            return vector_store
        embeddings = AzureOpenAIEmbeddings(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment="bbik-embedding-small",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version="2024-08-01-preview",
        )
        vector_store.add_documents(documents)
        print("[DEBUG] Added new documents to existing vector store.")
        return vector_store
    except Exception as e:
        st.error(f"Error adding documents to vector store: {e}")
        traceback.print_exc()
        return vector_store

# โหลดข้อมูล cache file
def load_data_sources():
    CACHE_FILE = Path("cached_documents.json")
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                documents = [
                    Document(page_content=item["page_content"], metadata=item["metadata"])
                    for item in data
                ]
                return documents
        except json.JSONDecodeError as e:
            st.error(f"Error reading cached JSON file: {e}")
            return []
        except Exception as e:
            st.error(f"Unexpected error loading cache file: {e}")
            return []
    else:
        st.error(f"Cache file not found at {CACHE_FILE}")
        return []

# Chunk เอกสาร: Recursive Chunking + Semantic Chunking (Hybrid Approach)
def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    splitted_docs = []
    for doc in docs:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            splitted_docs.append(Document(page_content=chunk, metadata=doc.metadata))

    try:
        embed_model = AzureOpenAIEmbeddings(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment="bbik-embedding-small",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version="2024-08-01-preview",
        )
        
        semantic_chunker = SemanticChunker(embed_model)
        final_chunks = []
        for doc in splitted_docs:
            chunks = semantic_chunker.split_text(doc.page_content)
            for chunk in chunks:
                final_chunks.append(Document(page_content=chunk, metadata=doc.metadata))
        return final_chunks

    except Exception as e:
        st.error(f"Error during semantic chunking: {e}")
        traceback.print_exc()
        return splitted_docs

# Debug vector_store
def debug_vector_store(vector_store):
    if vector_store:
        print("[DEBUG] Vector store contains the following documents (top 10):")
        docs = vector_store.similarity_search("", k=10)
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source','Unknown')
            print(f"{i+1}) Source: {source}")
    else:
        print("[DEBUG] Vector store is empty.")

# Process ไฟล์แต่ละประเภท
def process_pdf(file_path):
    try:
        text = pymupdf4llm.to_markdown(file_path)
        if not text.strip():
            raise ValueError("Empty text layer, attempting OCR...")
        return [Document(page_content=text, metadata={"source": Path(file_path).name})]
    except Exception:
        pdf = pymupdf4llm.open(file_path)
        ocr_text = ""
        for page in pdf:
            img = page.get_pixmap()
            pil_img = Image.frombytes("RGB", [img.width, img.height], img.samples)
            ocr_text += pytesseract.image_to_string(pil_img, lang="eng+tha")
        if not ocr_text.strip():
            raise ValueError("Failed to extract text from PDF using OCR.")
        return [Document(page_content=ocr_text, metadata={"source": Path(file_path).name})]

def process_docx(file_path):
    try:
        text = docx2txt.process(file_path)
        if not text.strip():
            raise ValueError("No text found in DOCX file.")
        return [Document(page_content=text, metadata={"source": Path(file_path).name})]
    except Exception as e:
        raise ValueError(f"Error processing DOCX file: {e}")

def process_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            raise ValueError("TXT file is empty.")
        return [Document(page_content=text, metadata={"source": Path(file_path).name})]
    except Exception as e:
        raise ValueError(f"Error processing TXT file: {e}")

def process_image(file_path):
    try:
        img = Image.open(file_path)
        extracted_text = pytesseract.image_to_string(img, lang="eng+tha")
        if extracted_text.strip():
            return [Document(page_content=extracted_text, metadata={"source": Path(file_path).name})]
        else:
            st.warning(f"No text extracted from {Path(file_path).name}.")
            return []
    except Exception as e:
        st.error(f"Error processing image {Path(file_path).name}: {e}")
        return []

def process_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        text = df.to_string(index=False)
        if not text.strip():
            raise ValueError("CSV file is empty.")
        return [Document(page_content=text, metadata={"source": Path(file_path).name})]
    except Exception as e:
        raise ValueError(f"Error processing CSV file: {e}")

def process_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        text = df.to_string(index=False)
        if not text.strip():
            raise ValueError("Excel file is empty.")
        return [Document(page_content=text, metadata={"source": Path(file_path).name})]
    except Exception as e:
        raise ValueError(f"Error processing Excel file: {e}")

# New: Process SQL file
def process_sql(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            raise ValueError("SQL file is empty.")
        return [Document(page_content=text, metadata={"source": Path(file_path).name})]
    except Exception as e:
        raise ValueError(f"Error processing SQL file: {e}")

# process_uploaded_files
def process_uploaded_files(uploaded_files):
    new_docs = []
    for uploaded_file in uploaded_files:
        temp_file_path = tempfile.mktemp(suffix=Path(uploaded_file.name).suffix)
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        try:
            file_type, _ = mimetypes.guess_type(uploaded_file.name)
            file_suffix = Path(uploaded_file.name).suffix.lower()
            processed_docs = []
            if file_suffix == ".sql":
                processed_docs = process_sql(temp_file_path)
            elif file_type == "application/pdf":
                processed_docs = process_pdf(temp_file_path)
            elif file_type == "text/plain":
                processed_docs = process_txt(temp_file_path)
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                processed_docs = process_docx(temp_file_path)
            elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
                processed_docs = process_image(temp_file_path)
            elif file_type == "text/csv":
                processed_docs = process_csv(temp_file_path)
            elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet","application/vnd.ms-excel"]:
                processed_docs = process_excel(temp_file_path)
            else:
                st.warning(f"Unsupported file type: {file_type}")

            for doc in processed_docs:
                doc.metadata["source"] = uploaded_file.name
            new_docs.extend(processed_docs)
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
            traceback.print_exc()
        finally:
            os.remove(temp_file_path)

    if new_docs:
        print(f"[DEBUG] New documents added: {[doc.metadata['source'] for doc in new_docs]}")
    return new_docs

# calculate_token_size 
def calculate_token_size(text, model_name="gpt-4"):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception:
        return None

# Combined prompt: Chat and SQL in one
def create_combined_chain(llm):
    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an AI assistant for Bluebik Company's HR department. "
            "Below is the JSON format of all relevant documents:\n\n"
            "```json\n"
            "{context}\n"
            "```\n\n"
            "Each document object has keys: 'source' (filename) and 'content'. "
            "Treat 'content' as the full text. "
            "You already have all the text; do NOT say you cannot open files.\n\n"
            "Let's think step by step: "
            "First, read the user's question carefully. Then answer in Thai with as much detail "
            "as needed. Your response can include plain text, LaTeX equations (using $$...$$) and Markdown tables. "
            "Please note that the response should be no more than 500 tokens."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    return create_stuff_documents_chain(llm, combined_prompt, document_variable_name="context")

# Unified response function
def get_combined_response(user_input: str):
    if "vector_store" not in st.session_state or not st.session_state.vector_store:
        return "Vector store is not initialized. Please upload relevant documents or rely on the existing cache."

    try:
        all_docs = list(st.session_state.vector_store.docstore._dict.values())
        target_file = None
        # Check if the user explicitly mentioned a file name
        for doc in all_docs:
            doc_name = doc.metadata.get("source", "").lower().strip()
            if doc_name and doc_name in user_input.lower():
                target_file = doc.metadata["source"]
                break

        if target_file:
            file_docs = [doc for doc in all_docs if doc.metadata.get("source", "").lower() == target_file.lower()]
            if not file_docs:
                return f"File {target_file} not found in vector store documents."
            mini_store = FAISS.from_documents(file_docs, st.session_state.vector_store.embeddings)
            retriever = mini_store.as_retriever(search_kwargs={"k": 10})
            search_context = retriever.get_relevant_documents(user_input)
        else:
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 10})
            search_context = retriever.get_relevant_documents(user_input)

        if not search_context:
            return "I couldn't find any relevant context. Could you please provide more details or upload the necessary document?"

        docs_json = {"documents": []}
        for doc in search_context:
            docs_json["documents"].append({
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content
            })

        json_doc = Document(
            page_content=json.dumps(docs_json, ensure_ascii=False, indent=2),
            metadata={"source": "aggregated_json"}
        )

        llm = create_chat_model()
        combined_chain = create_combined_chain(llm)
        outputs = combined_chain.invoke({
            "input": user_input,
            "chat_history": st.session_state.chat_history,
            "context": [json_doc]
        })

        response_text = str(outputs)
        return response_text.strip()
    except Exception as e:
        traceback.print_exc()
        return f"An error occurred: {str(e)}"

# render chat history
def render_chat_history():
    for msg in st.session_state.chat_history:
        if isinstance(msg, AIMessage):
            with st.chat_message("AI"):
                st.markdown(msg.content)
        else:
            with st.chat_message("Human"):
                st.write(msg.content)

# MAIN APP
def main():
    st.title("Chat with Bluebik HR & SQL Assistant👩🏻‍💻")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if st.session_state.vector_store is None:
        cached_docs = load_data_sources()
        if cached_docs:
            chunked = chunk_documents(cached_docs)
            st.session_state.vector_store = setup_vectorstore(chunked)
            if st.session_state.vector_store:
                print("[DEBUG] Vector store created from cached docs.")
        else:
            st.warning("No documents found from cache. Please upload files to continue.")

    st.sidebar.header("อัพโหลดไฟล์เพิ่มเติม📁")
    uploaded_files = st.sidebar.file_uploader(
        label="อัพโหลด (txt, pdf, docx, csv, xlsx, png, jpeg, sql)",
        type=["txt", "pdf", "docx", "csv", "xlsx", "png", "jpeg", "sql", "jpg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.sidebar:
            st.write("ไฟล์ที่อัพโหลด:")
            for uf in uploaded_files:
                st.write(uf.name)

            with st.spinner("กำลังประมวลผลไฟล์อัพโหลด..."):
                uploaded_docs = process_uploaded_files(uploaded_files)
                if uploaded_docs:
                    chunked_up = chunk_documents(uploaded_docs)
                    if st.session_state.vector_store is None:
                        st.session_state.vector_store = setup_vectorstore(chunked_up)
                    else:
                        st.session_state.vector_store = add_documents_to_vectorstore(
                            st.session_state.vector_store, chunked_up
                        )
                    st.success("อัพโหลดไฟล์สำเร็จแล้ว!")
                    debug_vector_store(st.session_state.vector_store)
                else:
                    st.warning("ไม่มีข้อความที่ถูกต้องที่แยกออกมาจากไฟล์ที่อัพโหลด")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="สวัสดี! ฉันสามารถช่วยอะไรคุณได้บ้างวันนี้?")]

    user_input = st.chat_input("ถามคำถามของคุณหรือระบุความต้องการ SQL...")
    if user_input and user_input.strip():
        with st.spinner("กำลังประมวลผลคำถามของคุณ..."):
            response = get_combined_response(user_input)
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=response))

    render_chat_history()

if __name__ == "__main__":
    main()
