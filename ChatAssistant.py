import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="SAM Chat Assistant", layout="wide")

# --- Styles ---
st.markdown("""
    <style>
    .chat-bubble {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 1rem;
        max-width: 80%;
    }
    .user-bubble {
        background-color: #d1e7dd;
        align-self: flex-end;
        margin-left: auto;
    }
    .bot-bubble {
        background-color: #f8d7da;
        align-self: flex-start;
        margin-right: auto;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    .session-button {
        width: 100%;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session Setup ---
if "all_sessions" not in st.session_state:
    st.session_state.all_sessions = {}

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())

# Ensure current session has history
if st.session_state.current_session_id not in st.session_state.all_sessions:
    st.session_state.all_sessions[st.session_state.current_session_id] = [
        ("system", "You are a helpful AI assistant. Please respond to the user's queries clearly and concisely.")
    ]

# Get current session chat
chat_history = st.session_state.all_sessions[st.session_state.current_session_id]

# --- Sidebar ---
with st.sidebar:
    st.title("Sessions")
    st.markdown("Click a session to view its chat:")

    # Display sessions
    for sid in st.session_state.all_sessions:
        label = f" {sid[:8]}"
        if st.button(label, key=f"btn_{sid}"):
            st.session_state.current_session_id = sid
            st.rerun()

    st.divider()
    st.subheader("🌣 Settings")
    temperature = st.slider("Model Temperature", 0.0, 1.0, 0.7, step=0.05)

    if st.button("New Chat"):
        st.session_state.current_session_id = str(uuid.uuid4())
        st.rerun()

    if st.button("🗑 Clear This Session"):
        st.session_state.all_sessions[st.session_state.current_session_id] = [
            ("system", "You are a helpful AI assistant. Please respond to the user's queries clearly and concisely.")
        ]
        st.rerun()

# --- Main Content ---
st.title("🗿 Sam LangChain Assistant with Memory")

llm = Ollama(model="deepseek-r1:1.5b", temperature=temperature)
output_parser = StrOutputParser()

st.markdown("### ♡ㅤ ⎙ㅤ ⌲ Conversation")
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for role, message in chat_history:
    if role == "user":
        st.markdown(f'<div class="chat-bubble user-bubble"><strong>You:</strong> {message}</div>', unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f'<div class="chat-bubble bot-bubble"><strong>Bot:</strong> {message}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Chat Input Form ---
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("⌯⌲⋆｡◛ ⊹ ࣪ ᡣ𐭩₊⋆ Type your message here", key="user_input")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    chat_history.append(("user", user_input))
    prompt = ChatPromptTemplate.from_messages(chat_history)
    chain = prompt | llm | output_parser

    with st.spinner("💭 Thinking..."):
        try:
            response = chain.invoke({})
            chat_history.append(("assistant", response))
        except Exception as e:
            chat_history.append(("assistant", f"⚠️ Error: {str(e)}"))
    st.rerun()
