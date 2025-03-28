from openai import OpenAI

client = OpenAI()

# Function to send the prompt to the OpenAI API
def gpt():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": 
                f"""
                """
             }
        ]
    )
    return response.choices[0].message.content


res = gpt()
print(res)