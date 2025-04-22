import streamlit as st
from agent import AgentChat, client
import sqlite3


# Initialize the SQLite database connection
db_file = "Data/maintenance.db"
if 'conn' not in st.session_state:
    st.session_state.conn = sqlite3.connect(db_file, check_same_thread=False)

# Initialize the Agent and store it in session state
if "agent" not in st.session_state:
    from agent import client
    st.session_state.agent = AgentChat(client, 
                                       st.session_state.conn, 
                                       'gpt-4o-mini')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit App
st.title("Chatbot Interface")

# Add a reset button
if st.button("Reset Conversation"):
    st.session_state.messages = []  # Clear the chat history
    st.session_state.agent.reset_conversation_history()  # Call the reset method on the agent

# Display the previous messages from the session messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# If the user sends a message
if prompt := st.chat_input("You:"):
    # Display the message sent by the user
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save the message sent by the user
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get the response from the agent
    with st.spinner('Thinking...'):
        text_response, boolean_agent = st.session_state.agent.excecute_agent(prompt)

    # Display the response from the agent
    with st.chat_message("assistant"):
        st.markdown(text_response)

    # Save the agent's response
    st.session_state.messages.append({"role": "assistant", "content": text_response})