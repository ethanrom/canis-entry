from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import openai
import os
import config

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

template = """You are an expert in data visualization who can create suitable visualizations to find the required information. You have access to a dataset (dataset.csv) and you are gievn a question. Generate a python code with st.altair_chart to find the answer.

First 3 rows of the dataset:
Name (English),Name (Chinese),Region of Focus,Language,Entity owner (English),Entity owner (Chinese),Parent entity (English),Parent entity (Chinese),X (Twitter) handle,X (Twitter) URL,X (Twitter) Follower #,Facebook page,Facebook URL,Facebook Follower #,Instragram page,Instagram URL,Instagram Follower #,Threads account,Threads URL,Threads Follower #,YouTube account,YouTube URL,YouTube Subscriber #,TikTok account,TikTok URL,TikTok Subscriber #
Yang Xinmeng (Abby Yang),杨欣萌,Anglosphere,English,China Media Group (CMG),中央广播电视总台,Central Publicity Department,中共中央宣传部,_bubblyabby_,https://twitter.com/_bubblyabby_,1678.00,itsAbby-103043374799622,https://www.facebook.com/itsAbby-103043374799622,1387432.00,_bubblyabby_,https://www.instagram.com/_bubblyabby_/,9507.00,_bubblyabby_,https://www.threads.net/@_bubblyabby_,197.00,itsAbby,https://www.youtube.com/itsAbby,4680.00,_bubblyabby_,https://www.tiktok.com/@_bubblyabby_,660.00
CGTN Culture Express,,Anglosphere,English,China Media Group (CMG),中央广播电视总台,Central Publicity Department,中共中央宣传部,_cultureexpress,https://twitter.com/_cultureexpress,2488.00,,,,_cultureexpress/,https://www.instagram.com/_cultureexpress/,635.00,,,,,,,,,

=============
Question:
{question}

Example:
import altair as alt
import pandas as pd
import streamlit as st
df = pd.read_csv('dataset.csv')
# your code here
st.altair_chart(chart, use_container_width=True)

Generated Python Code:"""

prompt = PromptTemplate(template=template, input_variables=["question"])

def generate_plot(question):
    llm = ChatOpenAI(temperature=0)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run(question=question)
    return response


retry_template = """Current code attempts to create a visualization of dataset.csv to meet the objective. but it has encounted the given error. provide a corrected code.

First 3 rows of the dataset:
Name (English),Name (Chinese),Region of Focus,Language,Entity owner (English),Entity owner (Chinese),Parent entity (English),Parent entity (Chinese),X (Twitter) handle,X (Twitter) URL,X (Twitter) Follower #,Facebook page,Facebook URL,Facebook Follower #,Instragram page,Instagram URL,Instagram Follower #,Threads account,Threads URL,Threads Follower #,YouTube account,YouTube URL,YouTube Subscriber #,TikTok account,TikTok URL,TikTok Subscriber #
Yang Xinmeng (Abby Yang),杨欣萌,Anglosphere,English,China Media Group (CMG),中央广播电视总台,Central Publicity Department,中共中央宣传部,_bubblyabby_,https://twitter.com/_bubblyabby_,1678.00,itsAbby-103043374799622,https://www.facebook.com/itsAbby-103043374799622,1387432.00,_bubblyabby_,https://www.instagram.com/_bubblyabby_/,9507.00,_bubblyabby_,https://www.threads.net/@_bubblyabby_,197.00,itsAbby,https://www.youtube.com/itsAbby,4680.00,_bubblyabby_,https://www.tiktok.com/@_bubblyabby_,660.00
CGTN Culture Express,,Anglosphere,English,China Media Group (CMG),中央广播电视总台,Central Publicity Department,中共中央宣传部,_cultureexpress,https://twitter.com/_cultureexpress,2488.00,,,,_cultureexpress/,https://www.instagram.com/_cultureexpress/,635.00,,,,,,,,,

=============
Objective: {question}

Current Code:
{error_code}

Error Message:
{error_message}

Corrected Code:"""

retry_prompt = PromptTemplate(template=retry_template, input_variables=["question", "error_message, error_code"])

def retry_generate_plot(question, error_message, error_code):
    llm = ChatOpenAI(temperature=0)
    llm_chain = LLMChain(prompt=retry_prompt, llm=llm)
    response = llm_chain.run(question=question, error_message=error_message, error_code=error_code)
    return response