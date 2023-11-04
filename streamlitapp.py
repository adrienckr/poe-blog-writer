import os 
from apikey import key 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import WebBaseLoader

os.environ['OPENAI_API_KEY'] = key

st.title('AD ENGINE')
url = st.text_input('Enter the URL here')

script_template = PromptTemplate(
    input_variables = ['web_research'], 
    template='write me a  ad script in style of warden belfort based on this web research:{web_research} ')

script_memory = ConversationBufferMemory(input_key='web_research', memory_key='chat_history')

llm = OpenAI(temperature=0.9) 
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

web_loader = WebBaseLoader(url)

if url: 
    web_research = web_loader.load()
    script = script_chain.run(web_research=str(web_research))

    st.write(script) 

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    with st.expander('Web Research'): 
        st.info(web_research)