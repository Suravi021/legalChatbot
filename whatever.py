from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.prompts.prompts import PromptTemplate
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.langchain import LangChainLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import os
import re
from string import Template
import json


print("Loading dependencies...")

# Define system prompt
system_prompt = """
You are a legal assistant specialized in contract drafting and analysis.
Follow this process when helping users:
1. First ask for the parties involved in the agreement
2. Inquire about the general scenario or purpose of the contract
3. Based on the information, identify the type of contract (e.g., employment, lease, sale, NDA)
4. Ask follow-up questions specific to that contract type
5. Provide relevant legal considerations for the identified contract type

Always maintain a professional, precise tone and ensure legal compliance.
"""


ollama_llm = OllamaLLM(
    model="llama3",
    temperature=0.2,
    system=system_prompt
)

llm = LangChainLLM(llm=ollama_llm)

embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

# Configure global settings
Settings.llm = llm
Settings.embed_model = embed_model

documents = SimpleDirectoryReader("documents").load_data()
print("Documents loaded successfully")

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
print("Vector index created for RAG")




conversation_template = PromptTemplate(
"""
    You are a legal assistant helping with contracts.
    
    Conversation history:
    {context}
    
    User's latest message: {question}
    
    Based on the conversation so far, respond helpfully about the contract. 
    If this is the start of the conversation, ask about the purpose of the agreement between the parties.
    """
)


def handle_conversation():
    context = ""
    
    print("Welcome to the Legal Contract Assistant! Type 'exit' to quit and generate a json file.")
    print("Bot: Hello! I'm your legal contract assistant. Let's start by discussing your contract needs. Who are the parties involved in this agreement?")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        context += f"\nUser: {user_input}. make your response as short as possible. Only enquire about the details do no suggest anything"
        
        use_rag = any(keyword in user_input.lower() for keyword in [
            "legal", "requirement", "clause", "provision", "standard", "example", "template"
        ])
        
        if use_rag and query_engine:
            rag_response = query_engine.query(user_input)
            rag_context = f"Based on relevant legal information: {str(rag_response)}\n\n"
            
            enhanced_prompt = conversation_template.format(
                context=context + "\n" + rag_context,
                question=user_input
            )
            response_text = llm.complete(enhanced_prompt).text
        else:
            formatted_prompt = conversation_template.format(
                context=context,
                question=user_input
            )
            response_text = llm.complete(formatted_prompt).text

        
        print("Bot:", response_text)
        context += f"\nAI: {response_text}"

    print("CONTEXTTT:",context)
    return context



def extract_json_from_text(text):
    """Find and return the first JSON block from a string."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group() if match else "{}"

def extract_fields_from_context(context_text: str):
    
    prompt = 
    response = llm.complete(prompt)
    text_output = response.text
    json_part = extract_json_from_text(text_output)
    


if __name__ == "__main__":
    context=handle_conversation()
    final_thing=extract_fields_from_context(context)
    print(" Extracted Fields:")
    print(json.dumps(final_thing, indent=2))  