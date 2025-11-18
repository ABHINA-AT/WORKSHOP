import gradio
from groq import Groq
client = Groq(
    api_key="",
)
def initialize_messages():
    return [{"role": "system",
             "content": """You are a skilled agronomists
              expertized in soil health,irrigation management
              and pest management."""}]
messages_prmt = initialize_messages()
print(type(messages_prmt))
def customLLMBot(user_input, history):
    global messages_prmt

    messages_prmt.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        messages=messages_prmt,
        model="llama-3.3-70b-versatile",
    )
    print(response)
    LLM_reply = response.choices[0].message.content
    messages_prmt.append({"role": "assistant", "content": LLM_reply})

    return LLM_reply
iface = gradio.ChatInterface(customLLMBot,
                     chatbot=gradio.Chatbot(height=300),
                     textbox=gradio.Textbox(placeholder="Ask me a question related to agriculture?"),
                     title="Agriculture ChatBot",
                     description="Chat bot for planting assistance",
                     theme="soft",
                     examples=["hi","Which is the best time to plant ginger", "how many times in a day a plant should be watered" ]
                     )
iface.launch(share=True)