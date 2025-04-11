from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt, PromptTemplate 
from llama_index.embeddings.langchain import LangchainEmbedding
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings

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

# Load documents to provide legal knowledge base
try:
    documents = SimpleDirectoryReader("documents").load_data()
    print("Documents loaded successfully")
except Exception as e:
    print(f"Error loading documents: {e}")
    documents = []

# Configure LLM
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

MODEL = "vishnun0027/Llama-3.2-1B-Instruct-Indian-Law"

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.2, "do_sample": True},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=MODEL,
    model_name=MODEL,
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16}
)
print("LLM initialized")

# Set up embeddings
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

# Configure global settings
Settings.llm = llm
Settings.embed_model = embed_model

# Create the index from documents
if documents:
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    print("Vector index created")
else:
    print("Warning: No documents loaded, proceeding with LLM-only mode")
    query_engine = None

# Create a conversation template using LlamaIndex's PromptTemplate
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
    
    print("Welcome to the Legal Contract Assistant! Type 'exit' to quit.")
    print("Bot: Hello! I'm your legal contract assistant. Let's start by discussing your contract needs. Who are the parties involved in this agreement?")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        # Update context with user input
        context += f"\nUser: {user_input}"
        
        # Use the LLM directly with our template
        formatted_prompt = conversation_template.format(
            context=context,
            question=user_input
        )
        response_obj = llm.complete(formatted_prompt)
        response_text = response_obj.text
        
        print("Bot:", response_text)
        context += f"\nAI: {response_text}"

    print(context)

if __name__ == "__main__":
    handle_conversation()