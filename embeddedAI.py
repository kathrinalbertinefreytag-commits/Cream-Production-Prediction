from openai import OpenAI

client = OpenAI()

prompt = input("Bitte gib deinen Prompt ein: ")

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)