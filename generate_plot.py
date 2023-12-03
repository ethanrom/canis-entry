from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import openai
import pandas as pd
import os
import config

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

def read_first_3_rows():
    dataset_path = "dataset.csv"
    try:
        df = pd.read_csv(dataset_path)
        first_3_rows = df.head(3).to_string(index=False)
    except FileNotFoundError:
        first_3_rows = "Error: Dataset file not found."

    return first_3_rows

def generate_plot(question):
    dataset_first_3_rows = read_first_3_rows()

    GENERATE_PLOT_TEMPLATE_PREFIX = """You are an expert in data visualization who can create suitable visualizations to find the required information. You have access to a dataset (dataset.csv) and you are gievn a question. Generate a python code with st.altair_chart to find the answer.
    First 3 rows of the dataset:"""

    DATASET = f"{dataset_first_3_rows}"

    GENERATE_PLOT_TEMPLATE_SUFIX = """=============
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

    template = GENERATE_PLOT_TEMPLATE_PREFIX + DATASET + GENERATE_PLOT_TEMPLATE_SUFIX
    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm = ChatOpenAI(temperature=0)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run(question=question)
    return response



def retry_generate_plot(question, error_message, error_code):

    dataset_first_3_rows = read_first_3_rows()
    RETRY_TEMPLATE_PREFIX = """Current code attempts to create a visualization of dataset.csv to meet the objective. but it has encounted the given error. provide a corrected code.

    First 3 rows of the dataset:"""
    DATASET = f"{dataset_first_3_rows}"


    RETRY_TEMPLATE_SUFIX = """=============
    Objective: {question}

    Current Code:
    {error_code}

    Error Message:
    {error_message}

    Corrected Code:"""

    retry_template = RETRY_TEMPLATE_PREFIX + DATASET + RETRY_TEMPLATE_SUFIX
    retry_prompt = PromptTemplate(template=retry_template, input_variables=["question", "error_message, error_code"])

    llm = ChatOpenAI(temperature=0)
    llm_chain = LLMChain(prompt=retry_prompt, llm=llm)
    response = llm_chain.run(question=question, error_message=error_message, error_code=error_code)
    return response