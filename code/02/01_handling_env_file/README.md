# Multi-Provider LLM API Integration Starter

This repository provides a minimal Python setup to interact with frontier LLM APIs, including:

- OpenAI (GPT-4)
- Anthropic (Claude)
- Google (Gemini)

It uses `.env` files for secure API key management and demonstrates how to load and access keys via Python.

---

## Features

- `.env` file integration using `python-dotenv`
- Key access via `os.getenv`
- Safe key display (first 5 chars only)
- Template code for future API usage

---

### Install Dependencies

```bash
pip install python-dotenv
```

### Create Your `.env` File

In the project root, create a file named `.env` and add:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=claude-xxxxxxxxxxxxxxxxxxxxxxxx
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Important:** Do not include quotes around values.

### Add `.env` to `.gitignore`

```bash
echo ".env" >> .gitignore
```

---

## Running the Example

```bash
python example_env_usage.py
```

You should see output like:

```sh
OpenAI API Key: sk-abc...
Anthropic API Key: claude-xyz...
Google API Key: AIzaS...
```

---

## Notes

- All API keys are expected to be active and valid.
- Do not expose `.env` contents or push them to any public repositories.
- You can now extend `example_env_usage.py` to include real API calls.
