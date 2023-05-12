import os 

from apikey import apikey 

import streamlit as st
from langchain.llms import OpenAI

#wikipedia
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey #Call the api

# App framework
st.title('🦜🔗 Youtube GPT')
prompt = st.text_input('Plug in your prompt here') 




#promt template

from langchain.prompts import PromptTemplate  

title_template = PromptTemplate(

    input_variables = ['topic'],
    template = 'write me a youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title','wikipedia_research'],
    template = 'write me a youtube video script based on this title TITLE : {title} while leveraging this wikipedia reserch:{wikipedia_research}'
)


# Memory 
from langchain.memory import ConversationBufferMemory


title_memory = ConversationBufferMemory(input_key = 'topic', memory_key = 'chat_history')
script_memory = ConversationBufferMemory(input_key = 'title', memory_key = 'chat_history')


#llms

from langchain.chains import LLMChain, SequentialChain   

llm = OpenAI(temperature=0.9)

title_chain = LLMChain(llm=llm,prompt=title_template, verbose = True, output_key = 'title', memory = title_memory)

script_chain = LLMChain(llm=llm,prompt=script_template, verbose = True, output_key = 'script',memory = script_memory)



wiki = WikipediaAPIWrapper()


#show stuff to the screen

if prompt:

    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)

    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(script)

    with st.expander('Title History'):

        st.info(title_memory.buffer)

    with st.expander('Script History'):

        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research history'):

        st.info(wiki_research)




