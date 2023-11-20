from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool
from langchain.chains import LLMMathChain
from chat_chain import get_chatresponse
import streamlit as st
import pandas as pd
import plotly.express as px
import config
import os

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

#agent2 = create_csv_agent(
#    OpenAI(temperature=0),
#    "dataset.csv",
#    verbose=True,
#    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#)

def csv_agnet(string):
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
        "dataset.csv",
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    ans = agent.run(string)
    return ans

def math_tool(string):
    llm = OpenAI(temperature=0)
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    res = llm_math_chain.run(string)
    return res

def load_data():
    df = pd.read_csv("dataset.csv", encoding="utf-8")
    return df

def plot_visualization(selected_option, x_column, y_column):
    df = load_data()

    if df.empty:
        return st.warning("The data is empty.")

    if x_column not in df.columns or y_column not in df.columns:
        return st.warning("Invalid columns selected.")

    if selected_option == "bar":
        fig = px.bar(df, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
    elif selected_option == "scatter":
        fig = px.scatter(df, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
    elif selected_option == "line":
        fig = px.line(df, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
    elif selected_option == "scatter_matrix":
        fig = px.scatter_matrix(df, dimensions=[x_column, y_column], title=f"Scatter Matrix: {x_column} vs {y_column}")
    elif selected_option == "box":
        fig = px.box(df, x=x_column, y=y_column, title=f"Box Plot: {x_column} vs {y_column}")
    elif selected_option == "heatmap":
        fig = px.imshow(df.pivot_table(index=x_column, columns=y_column, aggfunc='size').fillna(0),
                        labels=dict(x=x_column, y=y_column),
                        title=f"Heatmap: {x_column} vs {y_column}")
    else:
        return st.warning("Please select a valid plot type.")

    return st.plotly_chart(fig)


def parsing_input(string):
    selected_option, x_column, y_column = string.split(",")
    return plot_visualization(selected_option, x_column, y_column)


zeroshot_tools = [
    Tool(
        name="answer_qa",
        func=csv_agnet,
        description="Use this tool to query the dataset. input to this tool should be a standalone question. Include the correct row titles that are needed. Example: How many rows are there in the dataset, which Facebook page has the highest Facebook Follower #",
        #return_direct=True,
    ),
    Tool(
        name="smalltalk",
        func=get_chatresponse,
        description="Use this tool to create a response to smalltalk user inputs. Input to this tool is the User Input you need to repond to. Example: Hello, Thank you",
        return_direct=True,
    ),
    Tool(
        name="create_simple_plot",
        func=parsing_input,
        description="""Use this tool to create x vs y plots. input is a comma seperated list of selected_option, x_column, y_column
        Example Inputs: 
        bar,Name (English),X (Twitter) Follower #
        line,Parent entity (English),Facebook Follower #
        line,Language,Facebook Follower #
        scatter_matrix,X (Twitter) Follower #,Instagram Follower #
        box,Region of Focus,TikTok Subscriber #
        heatmap,Language,Region of Focus
        These examples are only to show the input format. you can decide plot type based on the user input.
        """,
        #return_direct=True,
    ),
    Tool(
        name="Calculator",
        func=math_tool,
        description="useful when you need to do calculations. Example input: 21^0.43"
    ),
]
