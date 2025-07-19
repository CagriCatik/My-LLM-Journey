# LLM API Integration for Engineers

This document serves as a comprehensive guide for Large Language Model (LLM) engineers focused on integrating frontier LLM APIs—OpenAI (GPT-4), Anthropic (Claude), and Google (Gemini)—into their engineering workflows. It expands upon foundational concepts and focuses on real-world application through API calls, multi-model interoperability, and environment setup best practices.

---

## 1. Prerequisites

Before proceeding, the engineer should have the following skills and setup completed:

### ✅ Prerequisite Knowledge

* Understanding of:

  * Tokenization and context windows.
  * Transformer architecture fundamentals.
  * Prompt engineering basics (system/user prompts, one-shot prompting).
* Practical experience using:

  * Six major frontier LLMs via UI.
  * OpenAI API (GPT-4.0) for streaming, markdown, JSON output, chaining calls.

### ✅ Environment

* `JupyterLab` installed and operational.
* `.env` file configured for secret management and `git`-ignored.
* OpenAI API key configured.
* Python coding proficiency for API interaction.

---

## 2. Objectives

The goals for this week are:

1. **Integrate Anthropic Claude API**
2. **Integrate Google Gemini API**
3. **Implement Interoperable Code Between Models**
4. **Prepare Environment for Multi-LLM Usage**

---

## 3. Claude (Anthropic) API Integration

### API Key Setup

1. Visit: [https://www.anthropic.com](https://www.anthropic.com)
2. Sign up and generate your API key.
3. Copy the key to your project’s `.env` file:

   ```
   ANTHROPIC_API_KEY=your_key_here
   ```

### Advantages

* Easy registration.
* Free credits offered upon first-time sign-up (as of transcript date).
* Similar UX to OpenAI.

### Sample Usage (Python)

```python
import os
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0.7,
    messages=[
        {"role": "user", "content": "Explain diffusion models in simple terms."}
    ]
)

print(response.content)
```

---

## 4. Gemini (Google) API Integration

### API Key Setup

1. Sign in to [Google Cloud Console](https://console.cloud.google.com)
2. Enable the **Generative Language API**.
3. Create a **service account** with appropriate roles.
4. Generate an **API key**.
5. Add to your `.env` file:

   ```
   GOOGLE_API_KEY=your_key_here
   ```

### Caveats

* Complex and non-intuitive interface.
* May involve navigating multiple permission screens.
* Failure-prone for first-timers.

### Workaround

* Optional integration. All examples can be executed with just OpenAI and Anthropic if Gemini setup is too problematic.

---

## 5. API Key Security

### Recommended Practice

* Store keys in a `.env` file located at your project root.
* Ensure the `.env` file is added to `.gitignore`.

### Alternative (Not Recommended)

* Hardcoding keys in Jupyter Notebook for quick testing:

  ```python
  API_KEY = "your_key_here"
  ```
* **Warning**: Never push hardcoded secrets to version control.

---

## 6. Multi-Model Interaction Example

This code illustrates switching between APIs in a unified function:

```python
def call_model(provider, prompt):
    if provider == "openai":
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        return openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content
    
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        ).content

    elif provider == "google":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text

    else:
        raise ValueError("Unsupported provider")
```

---

## 7. Roadmap Preview

|  | Focus Area                             |
| ---- | -------------------------------------- |
| 1    | Transformer basics, GPT-4 usage        |
| 2    | Multi-API integration (Claude, Gemini) |
| 3    | UI Development, Agents, Multimodal     |
| 4    | Hugging Face & Open Source LLMs        |
| 5    | Model Selection & Code Generation      |
| 6    | Retrieval-Augmented Generation (RAG)   |
| 7    | Fine-tuning (Frontier & OSS)           |
| 8    | Capstone Projects & Demo               |

---

## 8. Next Steps

* Launch your development interface:

  ```
  jupyter lab
  ```
* Verify `.env` file includes all necessary keys.
* Begin testing Claude and Gemini calls with aligned prompts.
* Proceed to UI prototyping and agent integration next week.

---

## 9. Final Notes

* Focus on `.env` hygiene to avoid secret leakage.
* Claude integration is straightforward.
* Gemini setup is tedious but non-blocking.
* Multi-provider code sets you up for future model orchestration.
