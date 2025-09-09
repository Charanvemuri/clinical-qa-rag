from langchain_core.prompts import ChatPromptTemplate

SYSTEM = (
    "You are a clinical QA assistant that answers using only the provided context. "
    "If the answer is not in the context, say you don't know. Provide concise, evidence-based answers with citations."
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer with numbered citations [1], [2], ...")
])
