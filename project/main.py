# This file contains the code for the FastAPI application that serves as the API endpoint for the model.
import pickle
from sentence_transformers import SentenceTransformer
from faiss import read_index
from transformers import TFAutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, confloat


class Content(BaseModel):
    content: str
    temperature: confloat(ge=0.0, le=2.0) = 0.6
    top_p: confloat(ge=0.0, le=1.0) = 0.7
    repetition_penalty: confloat(ge=-2.0, le=2.0) = 1.2


model = TFAutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
index = read_index('project/vector_base.idx')
with open('project/texts_base.pkl', 'rb') as f:
    texts = pickle.load(f)

MAX_TOKENS = 1024


async def get_context(prompt: str) -> str:
    "This function retrieves the most similar context to the prompt from the base of texts."
    vector_prompt = encoder.encode(prompt, convert_to_numpy=True)
    vector_prompt = vector_prompt.reshape((1, -1))
    _, i = index.search(vector_prompt, 5)
    context = '\n\n'.join(texts[j] for j in i[0])
    return context


async def generate_text(content: Content) -> dict:
    "This function generates a response to a user query based on the provided context and user query."
    prompt = content.dict()['content'].strip()
    template = (
        f"Context: The capital of France is Paris. It has been the country's political and cultural center for centuries."
        f" Paris is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral."
        f" It is also the largest city in France and a hub for art, fashion, and history.\n\n"
        "Instruction: Provide a clear and accurate response to the user query below, using the information from the context above.\n\n"
        f"User query: What is the capital of France?\n\n"
        "Response: The capital of France is Paris. I know this because Paris has been the capital for centuries, serving"
        " as the political and cultural heart of the country. Its rich history and famous landmarks like the "
        "Eiffel Tower and the Louvre Museum make it one of the most well-known cities globally.\n\n\n"
        f"Context: The current president of the United States is Joe Biden. He became the 46th president after the 2020 election."
        f" Biden was inaugurated on January 20, 2021, and his administration focuses on addressing issues like climate change, healthcare, and economic recovery."
        f" His leadership emphasizes unity, rebuilding infrastructure, and tackling the COVID-19 pandemic.\n\n"
        "Instruction: Provide a clear and accurate response to the user query below, using the information from the context above.\n\n"
        f"User query: Who is the current president of the United States?\n\n"
        "Response: The current president of the United States is Joe Biden. He assumed office after winning the 2020 election,"
        " and his administration focuses on addressing key national and global challenges such as climate change and economic recovery."
        " He was inaugurated in January 2021 and is working towards unity and improving the well-being of American citizens.\n\n\n"
        f"Context: \n\n"
        "Instruction: Provide a clear and accurate response to the user query below, using the information from the context above.\n\n"
        f"User query: {prompt}\n\n"
        "Response:")
    inputs = tokenizer(template, return_tensors="tf")
    num_tokens_without_context = inputs["input_ids"].shape[1]
    if num_tokens_without_context > MAX_TOKENS:
        raise HTTPException(status_code=400, detail="Prompt and template exceed the model's context window.")
    available_tokens = MAX_TOKENS - num_tokens_without_context
    if available_tokens > 0:
        context = await get_context(prompt)
        context_tokens = tokenizer(context, max_length=available_tokens, truncation=True)["input_ids"]
        context = tokenizer.decode(context_tokens)
        final_prompt = (
            f"Context: The capital of France is Paris. It has been the country's political and cultural center for centuries."
            f" Paris is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and the Notre-Dame Cathedral."
            f" It is also the largest city in France and a hub for art, fashion, and history.\n\n"
            "Instruction: Provide a clear and accurate response to the user query below, using the information from the context above.\n\n"
            f"User query: What is the capital of France?\n\n"
            "Response: The capital of France is Paris. I know this because Paris has been the capital for centuries, serving"
            " as the political and cultural heart of the country. Its rich history and famous landmarks like the "
            "Eiffel Tower and the Louvre Museum make it one of the most well-known cities globally.\n\n\n"
            f"Context: The current president of the United States is Joe Biden. He became the 46th president after the 2020 election."
            f" Biden was inaugurated on January 20, 2021, and his administration focuses on addressing issues like climate change, healthcare, and economic recovery."
            f" His leadership emphasizes unity, rebuilding infrastructure, and tackling the COVID-19 pandemic.\n\n"
            "Instruction: Provide a clear and accurate response to the user query below, using the information from the context above.\n\n"
            f"User query: Who is the current president of the United States?\n\n"
            "Response: The current president of the United States is Joe Biden. He assumed office after winning the 2020 election,"
            " and his administration focuses on addressing key national and global challenges such as climate change and economic recovery."
            " He was inaugurated in January 2021 and is working towards unity and improving the well-being of American citizens.\n\n\n"
            f"Context: {context}\n\n"
            "Instruction: Provide a clear and accurate response to the user query below, using the information from the context above.\n\n"
            f"User query: {prompt}\n\n"
            "Response:")
        inputs = tokenizer(final_prompt, return_tensors="tf")

    top_p = content.dict()['top_p']
    temperature = content.dict()['temperature']
    repetition_penalty = content.dict()['repetition_penalty']
    output = model.generate(**inputs,
                            max_length=1500,
                            do_sample=True,
                            top_p=top_p,
                            temperature=temperature,
                            repetition_penalty=repetition_penalty,
                            )
    return {'response': tokenizer.decode(output[0])}


app = FastAPI()


@app.post("/generate/")
async def generate(content: Content):
    return await generate_text(content)
