import json
import re
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.langchain import LangChainLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# Define prompts
system_prompt = """You are a legal assistant. Extract contract data as JSON."""
# query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

# Initialize HuggingFace LLM (TinyLlama)
# llm = HuggingFaceLLM(
#     context_window=4096,
#     max_new_tokens=256,
#     generate_kwargs={"temperature": 0.2, "do_sample": True},
#     system_prompt=system_prompt,
#     query_wrapper_prompt=query_wrapper_prompt,
#     tokenizer_name="vishnun0027/Llama-3.2-1B-Instruct-Indian-Law",
#     model_name="vishnun0027/Llama-3.2-1B-Instruct-Indian-Law",
#     device_map="auto",
#     model_kwargs={"torch_dtype": torch.float16}
# )

# ollama_llm = OllamaLLM(
#     model="llama3",
#     temperature=0.2,
#     system=system_prompt
# )

# llm = LangChainLLM(llm=ollama_llm)

# def extract_json_from_text(text):
#     """Find and return the first JSON block from a string."""
#     match = re.search(r"\{.*\}", text, re.DOTALL)
#     return match.group() if match else "{}"

# def extract_fields_from_context(context_text: str):
#     prompt = f"""
# ### Instruction:
# Extract the following fields from the text below and return them as a JSON object.

# Fields:
# - parties
# - effective_date
# - agreement_duration
# - payment_terms
# - special_conditions

# If a field is missing, use null.

# ### Text:
# \"\"\"{context_text}\"\"\"
# ### Response:
# """
#     response = llm.complete(prompt)
#     text_output = response.text
#     json_part = extract_json_from_text(text_output)
#     return json.loads(json_part)

# # Example input from user
# user_input = """
# clients- alice&co, J&J, work on baby food technolgy, cereal,
# granola bars, diapers and other products etc,
# j&J will take care of manufacture and design
# alice&co will have packing rights.
# payment- 25% profit will go to J&J and 75% to alice and co
# timeline of the project- 13th MAY 2024 to 20th july 2026
# """

# parsed_result = extract_fields_from_context(user_input)

# print(" Extracted Fields:")
# print(json.dumps(parsed_result, indent=2))