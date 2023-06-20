import requests
import json
import aiohttp
import asyncio
import datetime

api_endpoint = "http://192.168.1.50:5000/api/v1/generate"
api_headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
    }
api_parameters = {
    "prompt": "",
    "stopping_strings": ['\n### Response:', '\n### Instruction:'],
    "add_bos_token": True,
    "ban_eos_token": False,
    "do_sample": False,
    "encoder_repetition_penalty": 1,
    "early_stopping": False,
    "length_penalty": 1,
    "min_length": 0,
    "max_new_tokens": 1500,
    "truncation_length": 2048,
    "no_repeat_ngram_size": 0,
    "num_beams": 1,
    "penalty_alpha": 0,
    "preset": "Chat",
    "seed": -1,
    "skip_special_tokens": True,
    "epsilon_cutoff": 0,
    "eta_cutoff": 0,
    "mirostat_mode": 0,
    "mirostat_tau": 5,
    "mirostat_eta": 0.2,
    "repetition_penalty": 1.18,
    "temperature": 0.7,
    "tfs": 0.8,
    "top_a": 0,
    "top_k": 0,
    "top_p": 1,
    "typical_p": 1
    }


with open('prompt.txt', 'r') as f:
    contents = f.read()
    
prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n" + contents + "\n\n### Response:\n"

api_parameters["prompt"] = prompt

async def get_answer_from_llm(seed):
    global api_parameters
    
    api_parameters["seed"] = seed
    
    parameter_string = json.dumps(api_parameters)
    async with aiohttp.ClientSession() as session:
        async with session.post(api_endpoint, headers=api_headers, data=parameter_string) as response:
            llm_response = await response.json()
            return llm_response['results'][0]['text']

       
async def main():
    for i in range(3):
        seed = datetime.datetime.now().timestamp()
        data = await get_answer_from_llm(seed)
        with open('results.txt', 'a+') as f:
            f.write("Seed: " + str(seed) + "\n" + data + "\n\n")

asyncio.run(main())
    