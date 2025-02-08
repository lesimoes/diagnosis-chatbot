import os
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from knowledge_base import load_knowledge_base, format_knowledge_base


# Load database
knowledge = load_knowledge_base("database.txt")
knowledge_context = format_knowledge_base(knowledge)

# Models and Pipeline
model_id="mistralai/Mistral-7B-Instruct-v0.3"
translation_model_id = "Helsinki-NLP/opus-mt-tc-big-en-pt"

# Chat parameters
first_ia_message = "Olá, quais são os seus sintomas?"
system_message = "You are a doctor who will help, based on the symptoms, and will give a diagnosis in Brazilian Portuguese. Your answer should be direct, simple and short, you can even ask a question to provide a more accurate answer. You should ask only about health. You should answer only questions about health."
text_placeholder = "Enter your text here."
text_waiting_ai_response = "Pensando..."
max_response_length = 256
reset_button_label = "Reset Chat History"
chatbot_title = "ChatBot Sintomas"
chatbot_description = f"* Um chatbot de sintomas que usa os modelos {model_id} e {translation_model_id}.* Lembre-se de não confiar nesse chatbot, para casos reais um médico deverá ser consultado."
temperature = 0.1

translation_pipeline = pipeline(
    "translation_en_to_pt",
    model=translation_model_id,
    token=os.getenv("HF_TOKEN")
)

def get_llm_hf_inference(model_id=model_id, max_new_tokens=128, temperature=temperature):
    llm = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        token = os.getenv("HF_TOKEN")
    )
    return llm


def translate_to_portuguese(text):
    translation = translation_pipeline(text)
    return translation[0]['translation_text']

# Configure the Streamlit app
st.set_page_config(page_title=chatbot_title, page_icon="🤗")
st.title(chatbot_title)
st.markdown(chatbot_description)

# Initialize session state for avatars
if "avatars" not in st.session_state:
    st.session_state.avatars = {'user': None, 'assistant': None}

# Initialize session state for user text input
if 'user_text' not in st.session_state:
    st.session_state.user_text = None

# Initialize session state for model parameters
if "max_response_length" not in st.session_state:
    st.session_state.max_response_length = max_response_length

# Sidebar for settings
with st.sidebar:
    st.header("System Settings")

    # AI Settings
    st.session_state.system_message = st.text_area(
        "System Message", value=system_message
    )
    st.session_state.starter_message = st.text_area(
        'First AI Message', value=first_ia_message
    )
    # Model Settings
    st.session_state.max_response_length = st.number_input(
        "Max Response Length", value=max_response_length
    )
    # Reset Chat History
    reset_history = st.button(reset_button_label)
    
# Initialize or reset chat history
if "chat_history" not in st.session_state or reset_history:
    st.session_state.chat_history = [{"role": "assistant", "content": st.session_state.starter_message}]

def get_response(system_message, chat_history, user_text, 
                 eos_token_id=['User'], max_new_tokens=max_response_length, get_llm_hf_kws={}):
    # Set up model with token and temperature
    hf = get_llm_hf_inference(max_new_tokens=max_new_tokens, temperature=temperature)

   
    # Create the prompt template
    prompt = PromptTemplate.from_template(
        (
            "[INST] {system_message}"
            "{knowledge_context}\n"
            "\nCurrent Conversation:\n{chat_history}\n\n"
            "\nUser: {user_text}.\n [/INST]"
            "\nAI:"
        )
    )

    # Include knowledge database
    chat = prompt | hf.bind(skip_prompt=True) | StrOutputParser(output_key='content')

    response = chat.invoke(input={
        "system_message": system_message,
        "knowledge_context": knowledge_context,
        "user_text": user_text,
        "chat_history": chat_history
    })
    response = response.split("AI:")[-1]
    response = translate_to_portuguese(response)
    chat_history.append({'role': 'user', 'content': user_text})
    chat_history.append({'role': 'assistant', 'content': response})
    return response, chat_history

# Chat interface
chat_interface = st.container(border=True)
with chat_interface:
    output_container = st.container()
    st.session_state.user_text = st.chat_input(placeholder=text_placeholder)
    
# Display chat messages
with output_container:
    for message in st.session_state.chat_history:
        if message['role'] == 'system':
            continue            
        with st.chat_message(message['role'], 
                             avatar=st.session_state['avatars'][message['role']]):
            st.markdown(message['content'])
            
 # User new text:
    if st.session_state.user_text:
        with st.chat_message("user", 
                             avatar=st.session_state.avatars['user']):
            st.markdown(st.session_state.user_text)
        with st.chat_message("assistant", 
                             avatar=st.session_state.avatars['assistant']):

            with st.spinner(text_waiting_ai_response):
                response, st.session_state.chat_history = get_response(
                    system_message=st.session_state.system_message, 
                    user_text=st.session_state.user_text,
                    chat_history=st.session_state.chat_history,
                    max_new_tokens=st.session_state.max_response_length,
                )
                st.markdown(response)