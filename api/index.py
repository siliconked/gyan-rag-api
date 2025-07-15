

# api/index.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# ✅ Modern imports (NO WARNINGS)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
import os
from dotenv import load_dotenv
load_dotenv()



# ----------- Load FAISS index -----------
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
index_path = os.path.join(current_dir, "gyan_faiss_index")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
# ----------- Load LLM -----------
llm = ChatOpenAI(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    openai_api_key=os.environ.get("OPENAI_API_KEY"),  # ✅ Together key
    openai_api_base="https://api.together.xyz/v1",     # ✅ Together endpoint
    temperature=0.7,
    max_tokens=512
)


# ----------- RAG Chain -----------
prompt = PromptTemplate.from_template("""
You are a smart and helpful assistant trained on real placement and internship experiences of NITK students.

Use the context below to answer the question. You may summarize, infer patterns, or estimate based on the context.
You can also use your own general knowledge to support your answer, but stay relevant to the NITK student experience.

Context:
{context}

Question:
{question}
""")

rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

followup_prompt = PromptTemplate.from_template("""
You are a career prep assistant.

Here's advice from seniors, 
{rag_output}

Now, based on this, generate a detailed preparation list, specific questions, the seniors' advice in 2-3 lines, and any company-specific strategies if and only if applicable.

Make sure the answer is detailed and covers all important points. Keep it concise.
""")

followup_chain = followup_prompt | llm | StrOutputParser()

# FastAPI App
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Gyan RAG API is live."}

@app.post("/ask")
async def ask_question(query: Query):
    try:
        rag_result = rag_chain.invoke(query.question)
        final_answer = followup_chain.invoke({"rag_output": rag_result})
        return {"answer": final_answer}
    except Exception as e:
        return {"error": str(e)}