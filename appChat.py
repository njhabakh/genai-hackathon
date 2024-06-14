import json
import os
import sys
import boto3
import streamlit as st

## We will be using Titan Embeddings Model To generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.chat_models.bedrock import BedrockChat

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store
# from langchain_community.vectorstores import FAISS
from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from pypdf import PdfReader
import chatbot_backend as demo

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)


#Extract PDF Data
def extract_pdf(filename):
    '''
    Extract all text of the PDF Data into one string
    args : 
        filname - name of the .pdf file
    returns:
        all_text - string output of all the text
    '''
    reader = PdfReader(filename)
    all_text = " "
    for i in range(0,len(reader.pages)):
        page = reader.pages[i]
        all_text = all_text + page.extract_text()
    return all_text

## Data ingestion
def data_ingestion(inp):
    loader=PyPDFDirectoryLoader(inp)
    documents=loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store

def get_vector_store(docs, inp):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local(f"faiss_index_{inp}")

def get_claude_llm():
    ##create the Anthropic Model
    llm=BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0",client=bedrock,
                model_kwargs={'max_tokens':1000})
    
    return llm

def get_llama2_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="meta.llama2-70b-chat-v1",client=bedrock,
                model_kwargs={'max_gen_len':200})
    
    return llm


prompt_template_chat = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

# Completeness check
# General Compliance
# Key Things

prompt_template_compliance = """
Imagine you are a compliance officer for a bank checking if policies and guidelines are being met.
Check the sections of the following document on whether the policies are being met.
<question>
{question}
</question

The following are the poilicies to be checked against:
<context>
{context}
</context

Provide the reason for non compliance with the corresponding section of the document 
and suggest edits to be made. Be as granular as possible. Provide just the summary of the non-compliant sections 
and a high level yes, no or partially compliant
in form of table with the section in one column, yes or no in the other column and the high level reason of non 
compliance or partial compliance in less than 10 words. 
Add the detailed summary under the table with the non compliant or partially compliant sections with quoted reference and 
suggested change. 
Please refer only to the document. 
Please be formal in your response. 
Please avoid any biases.
Assistant:"""

PROMPT1 = PromptTemplate(
    template=prompt_template_compliance, input_variables=["context", "question"]
)

PROMPT2 = PromptTemplate(
    template=prompt_template_chat, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query, PROMPT):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    print(query)
    answer=qa({"query":query})
    return answer['result']


def main():
    st.set_page_config("Team LLM")

    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
    tab1, tab2, tab3 = st.tabs(["Compi-Ease", "Compliance-Bot", "Regu-sync"])
    with tab1:
        # user_question = st.text_input("Ask a Question from the PDF Files") 
        # if uploaded_file is not None:
        #     user_question = extract_pdf(uploaded_file)

        # with st.sidebar:
        #     st.title("Update Or Create Vector Store:")
            
        #     if st.button("Vectors Update Guidelines"):
        #         with st.spinner("Processing..."):
        #             docs = data_ingestion('guidelines')
        #             get_vector_store(docs, 'guidelines')
        #             st.success("Done")

        # option = st.selectbox("Select the guidelines",
        #         ("EBA", "FINRA", "Regulatory", "Audit", "Legal", "Governance"))
        
        # if option == "EBA":
        #     with st.spinner("Processing..."):
        #         faiss_index = FAISS.load_local("VectorDB/faiss_index_eba", bedrock_embeddings, allow_dangerous_deserialization=True)
        #         llm=get_claude_llm()
        #         if user_question is not None:
        #         #document = extract_pdf(uploaded_file)
        #             st.write(get_response_llm(llm,faiss_index,user_question, PROMPT1))
        #             st.success("Done")
        
        # if st.button("EBA"):
        #     with st.spinner("Processing..."):
        #         faiss_index = FAISS.load_local("faiss_index_guidelines", bedrock_embeddings, allow_dangerous_deserialization=True)
        #         llm=get_claude_llm()
                
        #         st.write(get_response_llm(llm,faiss_index,user_question, PROMPT1))
        #         st.success("Done")

        # if st.button("FINRA"):
        #     with st.spinner("Processing..."):
        #         faiss_index = FAISS.load_local("faiss_index_guidelines", bedrock_embeddings, allow_dangerous_deserialization=True)
        #         llm=get_claude_llm()
                
        #         st.write(get_response_llm(llm,faiss_index,user_question, PROMPT1))
        st.success("Done")

    with tab2:
        #st.set_page_config(page_title='Ask the Document')
        st.title('Q&A from the Document')

        #if uploaded_file is not None:
            #document = extract_pdf(uploaded_file)


        # File upload
        #uploaded_file = st.file_uploader('Upload an article', type='txt')
        # Query text
        query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

        # Form input and query
        result = []
        with st.form('myform', clear_on_submit=True):
            #openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
            submitted = st.form_submit_button('Submit', disabled=not(uploaded_file))
            #if submitted and openai_api_key.startswith('sk-'):
            if submitted:
                with st.spinner('Loading...'):
                    llm=get_claude_llm()
                    if uploaded_file is not None:
                        document = extract_pdf(uploaded_file)
                        print(document)
                        response = demo.generate_response(document, llm, query_text)
                        result.append(response)
                    #del openai_api_key

        if len(result):
            st.info(response)
        #parameters
        
        st.title("Compliance Q&A ") # title

        # add langchain memory to session state
        if 'memory' not in st.session_state:
            st.session_state.memory = demo.demo_memory()

        # add chat history to session
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # render chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["text"])

        # input text box for chatbot
        input_text = st.chat_input("Type your question here")
        

        if input_text:
            with st.chat_message("user"):
                st.markdown(input_text)

            # Append user input to chat history
            st.session_state.chat_history.append({"role":"user", "text":input_text})

            # Generate chat response using the chatbot instance
            chat_response = demo.demo_conversation(input_text=input_text, memory=st.session_state.memory)

            # Display the chat response
            with st.chat_message("assistant"):
                st.markdown(chat_response["response"])

            # Append assistant's response to chat history
            st.session_state.chat_history.append({"role":"assistant", "text":chat_response["response"]})
            
                
        


    with tab3:
        st.header("Regu-sinc")
        st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

    


if __name__ == "__main__":
    main()














