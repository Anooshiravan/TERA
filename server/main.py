print ("--- TERA FastAPI server ---")

# ----------------------------------
# Loading transformers and libraries
# ----------------------------------
print ("Loading libraries...")
import torch
import os
import time
import math
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPTNeoForCausalLM, GPTJForCausalLM, AutoTokenizer
app = FastAPI()

# Setting number of CPU threads
print ("Setting the number of CPU threads to", os.cpu_count()-1)
torch.set_num_threads(os.cpu_count()-1)

# -----------------
# Loading GPT model
# -----------------
start_time = time.time()

if torch.cuda.is_available():
    if torch.cuda.get_device_properties(0).total_memory > 14147352576:
        model_name = 'EleutherAI/gpt-j-6B'
        print ("CUDA above 14GB. Loading model", model_name, "...")
        model = GPTJForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
    else:
        # Manually select a model based on GPU memory
        model_name = 'EleutherAI/gpt-neo-125M'
        # model_name = 'EleutherAI/gpt-neo-1.3B
        # model_name = 'EleutherAI/gpt-neo-2.7B
        print ("CUDA below 14GB. Loading model", model_name, "...")
        model = GPTNeoForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
else:
    model_name = 'EleutherAI/gpt-neo-125M'
    print ("CUDA not found. Loading model on CPU", model_name, "...")
    model = GPTNeoForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    
# Using auto tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
print ("Model took", math.floor(time.time() - start_time), "seconds to load.")


# -------------------
# Initial server test
# -------------------
start_time = time.time()

prompt = "Hello, my name is Tera and"
if torch.cuda.is_available():
    input_ids = tokenizer.encode(str(prompt), return_tensors='pt').cuda()
else:
    input_ids = tokenizer.encode(str(prompt), return_tensors='pt')

output = model.generate(
    input_ids,
    do_sample=True,
    max_length=32,
    # top_p=0.9,
    # top_k=0,
    temperature=0.8,
)
print('-------------------------------')
print('Test response time:', math.floor(time.time() - start_time))
print('Test response:',tokenizer.decode(output[0]))
print('-------------------------------')

# --------------
# API Functions
# --------------
def generator(input):
    start_time = time.time()

    if torch.cuda.is_available():
        input_ids = tokenizer.encode(str(input.prompt), return_tensors='pt').cuda()
    else:
        input_ids = tokenizer.encode(str(input.prompt), return_tensors='pt')

    token_count = input_ids.size(dim=1)

    # Limit token numbers
    # if token_count + input.generate_tokens_limit > 2048:
    #     raise Exception(f"This model can't generate more then 2048 tokens, you passed {token_count} "+
    #         f"input tokens and requested to generate {input.generate_tokens_limit} tokens") 
    
    output = model.generate(
        input_ids,
        do_sample=True,
        max_length=token_count + input.tokens,
        top_p=input.top_p,
        top_k=input.top_k,
        temperature=input.temperature,
    )

    resp = tokenizer.decode(output[0], skip_special_tokens=True)
    print('Tokenizer response time:', math.floor(time.time() - start_time))
    return resp

# Input class, defines POST request body
class Input(BaseModel):
    tokens: int = 64
    top_p: float = 0.7
    top_k: float = 0
    temperature: float = 1.0
    prompt: str
    
# POST Request
@app.post("/gpt/")
async def gpt(input: Input):
    try:
        output = generator(input)
        return {"Output": output}
    except Exception as e:
        return {"Error": str(e)}

# GET Request    
@app.get("/gpt/")
async def gpt():
        return {"Hello, TERA FastAPI is listening."}
    
# To run this server: $>uvicorn main:app