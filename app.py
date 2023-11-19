import streamlit as st
from streamlit_option_menu import option_menu
from config import set_openai_api_key
from memory import memory_storage
from agent_chain import get_agent_chain
from default_text import default_text4
from generate_plot import generate_plot, retry_generate_plot
from markup import app_intro, how_use_intro


if 'error' not in st.session_state:
    st.session_state['error'] = []

def tab1():

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("image.jpg", use_column_width=True)
    with col2:
        st.markdown(app_intro(), unsafe_allow_html=True)
    st.markdown(how_use_intro(),unsafe_allow_html=True) 


    github_link = '[<img src="https://badgen.net/badge/icon/github?icon=github&label">](https://github.com/ethanrom)'
    huggingface_link = '[<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">](https://huggingface.co/ethanrom)'

    st.write(github_link + '&nbsp;&nbsp;&nbsp;' + huggingface_link, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 14px; color: #777;'>Disclaimer: This app is a proof-of-concept and may not be suitable for real-world legal or policy decisions.</p>", unsafe_allow_html=True)


def tab2():

    openai_api_key = st.text_input("Enter your OpenAI API key:", type='password')

    st.header("🗣️ Chat")

    for i, msg in enumerate(memory_storage.messages):
        name = "user" if i % 2 == 0 else "assistant"
        st.chat_message(name).markdown(msg.content)

    if user_input := st.chat_input("User Input"):

        if openai_api_key:
            set_openai_api_key(openai_api_key)

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.spinner("Generating Response..."):

                with st.chat_message("assistant"):
                    zeroshot_agent_chain = get_agent_chain()
                    response = zeroshot_agent_chain({"input": user_input})

                    answer = response['output']
                    st.markdown(answer)
        else:
            st.warning("Please Enter Your Openai API Key First")
    
    if st.sidebar.button("Clear Chat History"):
        memory_storage.clear()


def tab3():
    openai_api_key = st.text_input("Enter your OpenAI API key:", type='password')

    st.header("📊 Data Visualization with NLP 🚀")
    st.markdown(
        """
        Explore your data like never before! Our Natural Language Processing (NLP)-powered tool transforms
        complex queries into stunning visualizations. Simply ask questions in plain language, and watch as
        insightful plots and charts come to life.
        """
    )

    question = st.text_area("🔍 Enter Your Query Here:", value=default_text4)

    result = None
    result2 = None

    if st.button("🚀 Generate Visualization"):

        if openai_api_key:
            set_openai_api_key(openai_api_key)

            with st.spinner('🤔 Thinking...'):
                result = generate_plot(question)
            st.session_state.generated_code = result

            st.subheader("👁️‍🗨️ Visualization")
            
            try:
                with st.spinner('📊 Generating Visualization...'):
                    exec(st.session_state.generated_code)
            except Exception as e:
                st.error(f"Error executing generated code: {str(e)}")
                st.session_state.error = str(e)
                
            st.markdown("____")
            
            with st.expander("👩‍💻 View The Generated Code"):
                st.code(result, language="python")
                    
        else:
            st.warning("Please Enter Your OpenAI API Key First")

    if st.session_state.error and st.button("Retry"):
        with st.spinner('🤔 Thinking...'):
            result2 = retry_generate_plot(question, st.session_state.error, st.session_state.generated_code)
        st.session_state.generated_code2 = result2        
        st.subheader("👁️‍🗨️ Visualization")

        try:
            with st.spinner('📊 Generating Visualization...'):
                exec(st.session_state.generated_code2)            
        except Exception as e:
            st.error(f"Error executing generated code: {str(e)}")

        st.code(result2, language="python") 

def main():
    st.set_page_config(page_title="Demo collection", page_icon=":memo:", layout="wide")
    tabs = ["Intro", "Data Visualization with NLP", "Chat"]

    with st.sidebar:

        current_tab = option_menu("Select a Tab", tabs, menu_icon="cast")

    tab_functions = {
    "Intro": tab1,
    "Data Visualization with NLP": tab3,
    "Chat": tab2,
    }

    if current_tab in tab_functions:
        tab_functions[current_tab]()

if __name__ == "__main__":
    main()