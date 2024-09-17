from typing import Set

from main_final2 import run_llm
import streamlit as st
from streamlit_chat import message

if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = ''


if 'llm' not in st.session_state:
    st.session_state['llm'] = ''

if 'db' not in st.session_state:
    st.session_state['db'] = ''

if 'loaded_db' not in st.session_state:
    st.session_state['loaded_db'] = '' 
    
if 'split_1' not in st.session_state:
    st.session_state['split_1'] = ''

if 'count' not in st.session_state:
    st.session_state['count'] = 0


st.header("Bec LS Website Bot")
st.write("Developed by Ravi Shankar Prasad, Data Scientist at Beckman Coulter Life Sciences.")

# st.write("You can ask me questions like below -- As I am GenAI enabled, you need not adhere to the exact form of the question as shown below but please ensure the context is captured")

# st.write("What are Beckman Cell Counters?")
# st.write("What are Beckman Sizers and Media Analyzers?")
# st.write("What is Beckman Flow Cytometry?")
# st.write("What are Beckman Automated Liquid Handling Solutions?")
# st.write("What are Beckman HIAC Liquid Particle Counters?")
# st.write("What are Beckman Microbioreactors?")
# st.write("What are Beckman Particle Size Analyzers?")
# st.write("What are Beckman Genomic Solutions?")
# st.write("How can I contact Beckman?")


prompt = st.text_input("Prompt", placeholder="Enter your prompt here..")
#prompt = st.chat_input("Prompt")
if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if prompt:
    with st.spinner("Generating response.."):
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )
        #st.write(generated_response)
        # sources = (
        #     [doc for doc in generated_response["context"]]
        # )

        # formatted_response = (
        #     f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        # )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(generated_response["answer"])
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["answer"]))


if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        st.session_state['count'] = st.session_state['count']+1
        message(user_query, is_user=True, key =  st.session_state['count'])
        message(generated_response)
