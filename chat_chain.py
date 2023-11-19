from langchain.memory import ReadOnlySharedMemory
from memory import memory
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import config
import os

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

temperature = 0



def get_chatresponse(string):

    CHAT_TEMPLATE = """You are an AI Assitant made for CANIS Data Visualization and Foreign Interference competition. offer brief and poliet smalltalk.

First 3 rows of the dataset:
Name (English),Name (Chinese),Region of Focus,Language,Entity owner (English),Entity owner (Chinese),Parent entity (English),Parent entity (Chinese),X (Twitter) handle,X (Twitter) URL,X (Twitter) Follower #,Facebook page,Facebook URL,Facebook Follower #,Instragram page,Instagram URL,Instagram Follower #,Threads account,Threads URL,Threads Follower #,YouTube account,YouTube URL,YouTube Subscriber #,TikTok account,TikTok URL,TikTok Subscriber #
Yang Xinmeng (Abby Yang),杨欣萌,Anglosphere,English,China Media Group (CMG),中央广播电视总台,Central Publicity Department,中共中央宣传部,_bubblyabby_,https://twitter.com/_bubblyabby_,1678.00,itsAbby-103043374799622,https://www.facebook.com/itsAbby-103043374799622,1387432.00,_bubblyabby_,https://www.instagram.com/_bubblyabby_/,9507.00,_bubblyabby_,https://www.threads.net/@_bubblyabby_,197.00,itsAbby,https://www.youtube.com/itsAbby,4680.00,_bubblyabby_,https://www.tiktok.com/@_bubblyabby_,660.00
CGTN Culture Express,,Anglosphere,English,China Media Group (CMG),中央广播电视总台,Central Publicity Department,中共中央宣传部,_cultureexpress,https://twitter.com/_cultureexpress,2488.00,,,,_cultureexpress/,https://www.instagram.com/_cultureexpress/,635.00,,,,,,,,,

====    
if the question is not related to the dataset, polietly inform you can only answer questions related to the dataset.
====
conversation history:
{chat_history}
====
User's New Input: {question}
====

AI:"""

    readonlymemory = ReadOnlySharedMemory(memory=memory)
    short_llm = ChatOpenAI(temperature=temperature)
    long_llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=temperature)
    llm = short_llm.with_fallbacks([long_llm])

    template = CHAT_TEMPLATE
    chat_prompt = PromptTemplate(template=template, input_variables=["question", "chat_history"])
    chat_chain = LLMChain(prompt=chat_prompt, llm=llm, memory=readonlymemory, verbose=True)
    chat_response = chat_chain.run(string)
    return chat_response