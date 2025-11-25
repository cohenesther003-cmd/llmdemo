from openai import OpenAI

client = OpenAI()

def say_hello():
    response = client.responses.create(
        model="gpt-4o-mini",
        input="Say hello to the world in one short sentence."
    )
    print(response.output[0].content[0].text)


if __name__ == "__main__":
    say_hello()