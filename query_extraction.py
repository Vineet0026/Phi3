from model_param import CFG, build_model
from process_output import process_llm_response
from langchain_community.llms import HuggingFacePipeline
import re
import ast
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from accelerate import Accelerator
from accelerate import load_checkpoint_and_dispatch
import json

with open('metadata.json') as f:
    d = json.load(f)


tokenizer, model = build_model(model_repo = CFG.model_name)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.bos_token_id
]

save_directory= "./model"
accelerator = Accelerator()

accelerator.save_model(model=model, save_directory=save_directory,max_shard_size="500MB")
device_map={'':'cpu'}
model = load_checkpoint_and_dispatch(
model, checkpoint="./model/", device_map=device_map, no_split_module_classes=['Block']
)

def generate_md(Question,query):
    messages = [{"role": "user", "content": f"{Question}{query}"}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=150)
    text = tokenizer.batch_decode(outputs)[0]
    text = process_llm_response(text)
    pattern = r'\["(.*?)",\s*{(?:\s*".*?":\s*".*?"\s*,?\s*)*}\]'
    match = re.search(pattern, text)
    if match:
        output_list = match.group(0)
        return(ast.literal_eval(output_list))
    else:
        print("No match found")
        return("[]")

def model_pipeline():
    pipe = pipeline(
        task = "text-generation",
        model = model,
        tokenizer = tokenizer,
        eos_token_id = terminators,
        do_sample = True,
        max_length = CFG.max_len,
        max_new_tokens = CFG.max_new_tokens,
        temperature = CFG.temperature,
        top_p = CFG.top_p,
        repetition_penalty = CFG.repetition_penalty,
    )

    llm = HuggingFacePipeline(pipeline = pipe)
    return llm
