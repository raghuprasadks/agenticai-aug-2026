import cohere

co = cohere.ClientV2(api_key="your cohere api key")

response = co.chat(
    messages=[
        {
            "role": "system",
            "content": "You  are a  helpful assitant"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "who are you"
                }
            ]
        },        
    ],
    thinking={
          "type": "enabled"
        },
    temperature=0.6,
    model="command-a-plus-05-2026",
)

#print(response)
print(response.message.content[0].text)