from ollama import chat
from typing import Generator, Dict, Any


def stream_llm_response(
    model: str,
    messages: list[Dict[str, str]],
    temperature: float = 0.7,
    num_predict: int = 512
) -> Generator[str, None, None]:
    """
    Stream text responses from an LLM using the Ollama chat API.

    Args:
        model: Name of the LLM model to use.
        messages: Sequence of message dicts with roles and content.
        temperature: Sampling temperature controlling randomness.
        num_predict: Maximum number of tokens to generate.

    Yields:
        Text chunks of LLM output.
    """
    for chunk in chat(
        model=model,
        messages=messages,
        stream=True,
        options={
            "temperature": temperature,
            "num_predict": num_predict
        }
    ):
        yield chunk["message"]["content"]


def explain_business_problem_suitability(question: str) -> None:
    """
    Print an explanation for deciding if a business problem suits an LLM solution.

    Args:
        question: The user prompt or question to ask the LLM.
    """
    messages = [{"role": "user", "content": question}]
    for content in stream_llm_response(
        model="llama3",
        messages=messages,
        temperature=0.7,
        num_predict=512
    ):
        print(content, end="", flush=True)


if __name__ == "__main__":
    prompt = (
        "Explain how to decide if a business problem is suitable for an LLM solution."
    )
    explain_business_problem_suitability(prompt)
