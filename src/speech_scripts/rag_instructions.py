#!/usr/bin/env python3
"""
rag_instructions.py  (FAISS version)

- Uses FAISS instead of Chroma as the vector store.
- Exposes: answer_query(user_query: str, chat_history: List[Tuple[str, str]]) -> str
"""

import os
from typing import List, Tuple

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


# ---------- OpenAI key setup ----------

# Your custom env var name
OPENAI_ENV_NAME = "OPENAI_API_KEY_WNI"

_openai_key = os.getenv(OPENAI_ENV_NAME)
if not _openai_key:
    raise RuntimeError(
        f"{OPENAI_ENV_NAME} is not set in the environment. "
        f"Run:  export {OPENAI_ENV_NAME}='sk-xxxx'"
    )

# LangChain / langchain-openai expects OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = _openai_key


# ---------- Paths for data & FAISS index ----------

HERE = os.path.dirname(os.path.abspath(__file__))
# Adjust if your txt file is elsewhere
TEXT_PATH = os.path.join(HERE, "floor_plan_text.txt")
FAISS_DIR = os.path.join(HERE, "faiss_openai_db")


def _load_text_chunks(path: str) -> List[str]:
    """Load and pre-process floor_plan_text.txt into chunks."""
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read().lower()

    # Split by blank lines
    texts = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

    # Remove duplicates while preserving order
    seen = set()
    unique_texts = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique_texts.append(t)

    print(f"📝 Loaded {len(unique_texts)} unique text chunks from {path}")
    return unique_texts


# ---------- Build / load FAISS index ----------

embedding_model = OpenAIEmbeddings()

if os.path.isdir(FAISS_DIR):
    # Load existing FAISS index
    print(f"📚 Loading existing FAISS index from {FAISS_DIR}")
    db = FAISS.load_local(
        FAISS_DIR,
        embedding_model,
        allow_dangerous_deserialization=True,  # needed with newer FAISS wrapper
    )
else:
    # Build new FAISS index from the txt file
    print(f"📚 FAISS index not found, creating new one at {FAISS_DIR}")
    texts = _load_text_chunks(TEXT_PATH)
    db = FAISS.from_texts(texts=texts, embedding=embedding_model)
    db.save_local(FAISS_DIR)
    print("📚 FAISS index created and saved")


# ---------- Retriever + QA chain ----------

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,          # number of final docs
        "fetch_k": 15,   # candidates
        "lambda_mult": 0.25,  # relevance vs diversity
    },
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    chain_type="stuff",
    retriever=retriever,
)

# ---------- Instruction prompt ----------

bvi_instruction = (
    "You are a friendly and helpful campus guide robot assisting blind or visually impaired students. "
    "Answer the following as if you are speaking to a blind person. "
    "Do not mention signs, colors, or visual markers. "
    "You may give instructions based on location and nearby rooms. "
    "If the user asks an explicit true/false question like 'is there a XXX', start your answer clearly with "
    "'Yes, there is...' or 'No, there is not...', followed by a simple explanation. "
    "If the question is about topics completely unrelated to campus (like weather, cooking, sports, etc.), "
    "politely say: 'I'm sorry, I can only assist with campus location and navigation questions.' "
    "If the user uses an abbreviation and it appears in the documents, always provide its details, even if brief. "
    "If you give an answer about a campus location or navigation question, you may ask: "
    "'Do you need me to guide you there?'\n"
    "Use ONLY the information provided in the retrieved documents. "
    "If the location is mentioned in any document, provide all available details clearly, even if brief. "
    "If there is truly no mention of the location at all, then and only then say: "
    "'I'm sorry, I do not have information about that.' "
    "Be friendly, concise, and avoid repeating irrelevant disclaimers."
)


# ---------- Public API ----------

def answer_query(user_query: str, chat_history: List[Tuple[str, str]]) -> str:
    """
    Main function used by your rag_service.
    :param user_query: current user question
    :param chat_history: list of (user, bot) turns
    :return: answer string
    """
    # Build history text
    history_text = ""
    for user_q, bot_a in chat_history:
        history_text += f"User: {user_q}\nAssistant: {bot_a}\n"

    # Retrieve documents
    retrieved_docs = retriever.get_relevant_documents(user_query)
    context_docs = "\n".join([f"- {doc.page_content}" for doc in retrieved_docs])

    full_query = (
        f"{bvi_instruction}\n\n"
        f"The following are the chat history between you and the user. Use this for reference purposes only. "
        f"Only refer and answer based on the history if really needed.\n"
        f"{history_text}\n\n"
        f"The following context documents were retrieved for your query. "
        f"If any of them mention the location or its abbreviation, use those details to answer. "
        f"Only say 'I'm sorry, I do not have information about that.' if none of the documents mention the location. "
        f"Answer based on one single retrieved doc among the 5 only. "
        f"Do not combine information across different docs. "
        f"Do not add or infer extra information not explicitly written in the chosen document.\n\n"
        f"Context:\n{context_docs}\n\n"
        f"User: {user_query}\nAssistant:"
    )

    response = qa_chain.invoke({"query": full_query})
    return response["result"]
