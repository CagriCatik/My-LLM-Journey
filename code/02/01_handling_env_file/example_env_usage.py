import os
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

def print_api_keys():
    print("OpenAI API Key: ", openai_key[:5] + "..." if openai_key else "Not found!")
    print("Anthropic API Key: ", anthropic_key[:5] + "..." if openai_key else "Not found!")
    print("Google API Key: ", google_key[:5] + "..." if openai_key else "Not found!")

if __name__ == "__main__":
    print_api_keys()