import streamlit as st
import streamlit as st
from langchain.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
from langchain_community.chat_models.bedrock import BedrockChat

bedrock = boto3.client(service_name="bedrock-runtime")
embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)


## author: Yuxuan Zeng
## contact: yuxuanzeng220@gmail.com

# # Set the Streamlit layout to wide
# st.set_page_config(layout="wide") 

# Page configuration
st.set_page_config(
    page_title="conformant",
    page_icon="🤓 ",
    layout="wide",
    # initial_sidebar_state="expanded"
    )

# Load CSS
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Page Title
st.image("title_image.PNG", use_column_width=True)
st.markdown(
    """ 
    <style>
    img{
    width: 30% !important;
    }
    </style> 
""", 
unsafe_allow_html=True
 )


# Tabs
# tabs = st.tabs([" 🔍 CompliCheck", " 🤖 Complibot ", " 🔄 ReguSync ", " 📰 ComplianceBrief ", " ℹ️ About ", " ❓ Help "])
tabs = st.tabs([" 🔍 CompliCheck", " 🤖 Complibot ", " 🔄 ReguSync ", " ℹ️ About ", " ❓ Help "])

# Import tab content
import tabs.complicheck as complicheck
import tabs.regusync as regusync
# import tabs.compliancebrief as compliancebrief
# import tabs.about as about
# import tabs.help as help



def get_claude_llm():
    ##create the Anthropic Model
    bedrock=boto3.client(service_name="bedrock-runtime")

    llm=BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0",client=bedrock,
                model_kwargs={'max_tokens':1000})
    return llm
    
# Tab content
with tabs[0]:
    complicheck.display()

with tabs[1]:
    # complibot.display()

    option = st.selectbox(
        "🏆 Choose the guideline to deep dive into",
        # ("FINRA", "BIA", "Regulatory", "External Audit", "Legal and Compliance")
        ("EBA",'Anti-Bribery', "Fraud", "Cyber  Security", "FINRA", "SEC"),
        index=None,
        placeholder="Select Guideline",
        key="abc"
    )

    query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.')
    llm = get_claude_llm()
    if option in ("EBA", "Fraud", "Cyber Security","Anti-Bribery") and len(query_text):
        with st.spinner("Processing..."):
            vector_db = FAISS.load_local(f"VectorDB\\faiss_index_{option.replace(' ','')}", bedrock_embeddings, allow_dangerous_deserialization=True)
            retriever = vector_db.as_retriever()
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            response = qa.run(query_text)
            st.write(response)
            st.success("Done")
    else:
        pass


with tabs[2]:
    regusync.display()

# with tabs[3]:
#     about.display()

# with tabs[4]:
#     help.display()

# Page footer
st.markdown(
    """
    <div style='position: fixed; bottom: 0; left: 0; width: 100%; background-color: #6baed6;  padding: 0px; text-align: left;'>
        <p style='color: white; font-size: 1em;'>&copy; 2024 Hackathon Team LLM</p>
    </div>
    """,
    unsafe_allow_html=True
)