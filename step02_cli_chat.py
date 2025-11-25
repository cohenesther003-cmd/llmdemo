from openai import OpenAI

SYSTEM_PROMPT = "You are a friendly tutor helping a beginner learn LLM workflows."

def chat():
    client = OpenAI()
    print("Type anything to chat. Enter 'exit' to quit.\n")
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    while True:
        user_text = input("You: ").strip()
        if not user_text or user_text.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return

        history.append({"role": "user", "content": user_text})
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": msg["role"], "content": msg["content"]}
                for msg in history
            ],
        )
        assistant_text = response.output[0].content[0].text
        print(f"Assistant: {assistant_text}\n")
        history.append({"role": "assistant", "content": assistant_text})

if __name__ == "__main__":
    chat()