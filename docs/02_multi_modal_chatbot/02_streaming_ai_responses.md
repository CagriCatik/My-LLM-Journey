# Streaming AI Responses

This document describes how to implement real-time LLM output (“streaming”) in Python using four providers: OpenAI, Anthropic (Claude), Google (Gemini), and local models via Ollama. It covers environment setup, key management (where required), code examples for both synchronous and streaming interactions, API parameter explanations, and best practices.

## Prerequisites

* Python 3.8+
* For OpenAI, Anthropic, Google:

  * Accounts and API keys
  * Installed SDKs

    ```bash
    pip install openai anthropic google-generativeai
    ```

* For Ollama (local models):

  * Docker (macOS or Linux) or Windows Subsystem for Linux (WSL)
  * Ollama daemon installed and running
  * Ollama Python client library

    ```bash
    pip install ollama
    ```

## Environment Configuration

1. Create a file named `.env` in your project root.
2. Populate it with provider keys (skip for Ollama):

   ```sh
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   GOOGLE_API_KEY=your_google_key
   ```

3. Load environment variables in code rather than embedding keys directly:

   ```python
   from dotenv import load_dotenv
   import os

   load_dotenv()
   os.environ["OPENAI_API_KEY"]   = os.getenv("OPENAI_API_KEY")
   os.environ["ANTHROPIC_API_KEY"]= os.getenv("ANTHROPIC_API_KEY")
   os.environ["GOOGLE_API_KEY"]   = os.getenv("GOOGLE_API_KEY")
   ```

## Common Message Structure

All four interfaces use a chat-style message list or a simple prompt. Each message is a dict with:

* `role`: one of `"system"`, `"user"` (and `"assistant"` in responses).
* `content`: the text to send.

Additional parameters may include:

* `model`: model identifier (e.g., `gpt-4`, `claude-3.5-sonnet`, `gemini-1.5-flash`, `llama3`)
* `temperature`: float in \[0,1] (higher = more creative)
* `max_tokens` / `n_predict` / `num_predict`: cap on output length
* `stream`: boolean for OpenAI; method switch or flag for others

## Connecting to APIs

### OpenAI

```python
import openai, os

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role":"system","content":"You are an assistant great at telling jokes."},
        {"role":"user","content":"Tell a lighthearted joke for data scientists."}
    ],
    temperature=0.7
)
print(response.choices[0].message.content)
```

### Anthropic (Claude)

```python
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import os

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

response = client.chat.completions.create(
    model="claude-3.5-sonnet",
    max_tokens_to_sample=256,
    prompt=(HUMAN_PROMPT +
            "Tell a lighthearted joke for data scientists.\n" +
            AI_PROMPT)
)
print(response.completion)
```

### Google (Gemini)

```python
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini = genai.models.TextGenerationModel.from_pretrained("gemini-1.5-flash")

response = gemini.generate(
    prompt="Tell a lighthearted joke for data scientists.",
    temperature=0.7
)
print(response.text)
```

### Ollama (Local Models)

1. Pull or install a model, e.g.:

   ```bash
   ollama pull llama3
   ```

2. Verify available models:

   ```bash
   ollama list
   ```

3. Functional interface:

   ```python
   from ollama import chat

   response = chat(
       model="llama3",
       messages=[{"role":"user","content":"Tell a lighthearted joke for data scientists."}],
       options={"temperature":0.7,"num_predict":256}
   )
   print(response["message"]["content"])
   ```

4. Client interface:

   ```python
   from ollama import Client

   client = Client()  # connects to localhost:11434
   response = client.chat(
       model="llama3",
       messages=[{"role":"user","content":"Tell a lighthearted joke for data scientists."}],
       options={"temperature":0.7,"num_predict":256}
   )
   print(response.message.content)
   ```

## Example: Joke Generation

| Provider          | Prompt                                          | Sample Output                                                                                                    |
| ----------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| GPT-3.5-turbo     | “Tell a lighthearted joke for data scientists.” | “Why did the data scientists break up with their computer? It couldn’t handle their complex relationship.”       |
| GPT-4             | same                                            | “Why did the data scientist break up with a statistician? Because she found him too mean.”                       |
| Claude 3.5 Sonnet | same                                            | “Why do data scientists break up with their significant other? Too much variance in the relationship.”           |
| Gemini 1.5 Flash  | same                                            | “Why did the data scientists break up with a statistician? Because they couldn’t see eye to eye on the p value.” |
| Llama3 (Ollama)   | same                                            | “Why did the data scientist break up with their database? It lost all their relationships.”                      |

Adjust `temperature` or `num_predict` for variation in randomness and length.

## Streaming Responses

### OpenAI Streaming

```python
import sys
from openai import ChatCompletion

stream = ChatCompletion.create(
    model="gpt-4",
    messages=messages,
    temperature=0.7,
    stream=True
)
for chunk in stream:
    sys.stdout.write(chunk.choices[0].delta.get("content",""))
    sys.stdout.flush()
```

### Anthropic Streaming

```python
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
with client.chat.completions.stream(
    model="claude-3.5-sonnet",
    max_tokens_to_sample=256,
    prompt=HUMAN_PROMPT + user_prompt + AI_PROMPT
) as stream:
    for event in stream:
        print(event.completion, end="", flush=True)
```

### Google Gemini

Streaming not supported; use synchronous only.

### Ollama Streaming

```python
from ollama import chat

for chunk in chat(
    model="llama3",
    messages=[{"role":"user","content":"Explain how to decide if a business problem is suitable for an LLM solution."}],
    stream=True,
    options={"temperature":0.7,"num_predict":512}
):
    print(chunk["message"]["content"], end="", flush=True)
```

Client-based equivalent:

```python
from ollama import Client

client = Client()
stream = client.chat(
    model="llama3",
    messages=[{"role":"user","content":"Explain how to decide if a business problem is suitable for an LLM solution."}],
    stream=True,
    options={"temperature":0.7,"num_predict":512}
)
for chunk in stream:
    print(chunk.message.content, end="", flush=True)
```

## Handling Markdown Streaming in Jupyter

```python
from IPython.display import clear_output, Markdown, display

buffer = ""
stream = openai.ChatCompletion.create(..., stream=True)
for chunk in stream:
    buffer += chunk.choices[0].delta.get("content","")
    clear_output(wait=True)
    display(Markdown(buffer))
```

Use the same pattern for Anthropic and Ollama streams.

## Comparison of APIs

| Feature                  | OpenAI                  | Anthropic (Claude)        | Google (Gemini)  | Ollama (Local)              |
| ------------------------ | ----------------------- | ------------------------- | ---------------- | --------------------------- |
| Endpoint style           | `ChatCompletion.create` | `chat.completions.create` | `model.generate` | `chat(...)` / `Client.chat` |
| Streaming activation     | `stream=True`           | `.stream(...)` method     | Not supported    | `stream=True` flag          |
| System message placement | In messages list        | Separate `prompt` prefix  | In constructor   | In messages list            |
| Token cap param          | `max_tokens`            | `max_tokens_to_sample`    | Implicit         | `num_predict`               |
| Context window           | 8K–128K tokens          | \~100K tokens             | 1 000 000 tokens | Model-dependent             |

## Best Practices

* **Key Management**: Store in `.env`, never hard-code (not required for Ollama).
* **Error Handling**: Catch provider-specific exceptions.
* **Rate Limits**: Respect external API limits; local models unrestricted.
* **Chunk Processing**: Accumulate partial content; avoid per-chunk newlines.
* **Token Budgeting**: Use `max_tokens`/`num_predict` to control length.
* **Reproducibility**: Set `temperature=0` for deterministic output.
* **Local vs Cloud**: Use Ollama for offline or privacy-sensitive tasks.

## Next Steps

* Enable model-to-model “self-play” conversations.
* Integrate multi-turn context management.
* Explore fine-tuning or embeddings pipelines.