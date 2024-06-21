import json
import os
import sys
import boto3
import streamlit as st
from dotenv import load_dotenv
from langchain.load import dumps, loads
import re
import pandas as pd
from pathlib import Path

load_dotenv()

## We will be using Titan Embeddings Model To generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.chat_models.bedrock import BedrockChat

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Vector Embedding And Vector Store
# from langchain_community.vectorstores import FAISS
from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from pypdf import PdfReader
from langchain.schema import Document

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

file_path = Path(__file__).parents[1]

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

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


def extract_pdf_docs(filename):
    '''
    Extract PDF Data as Docs per page
    args : 
        filname - name of the .pdf file
    returns:
        docs - Document output per page of the pdf file
    '''
    reader = PdfReader(filename)
    docs = []
    for i in range(0,len(reader.pages)):
        page = reader.pages[i]
        docs.append(Document(page_content=page.extract_text()))
    return docs


## Data ingestion
def data_ingestion(file_name):
    '''
    Extract PDF files from a directory and returns chunked docs
    args : 
        filname - name of the .pdf file
    returns:
        docs - Document output per chunk of the pdf files
    '''
    loader=PyPDFLoader(file_path /"guidelines" / file_name)
    documents=loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                                 chunk_overlap=100)
    
    docs=text_splitter.split_documents(documents)
    return docs

def save_vector_store(docs, inp):
    '''
    Extract PDF files from a directory and returns chunked docs
    args : 
        docs - Document input of the pdf file or files
        inp - directory location based on guideline
    returns:
    '''
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local(file_path/"VectorDB"/f"faiss_index_{inp}")

def get_response_llm(llm,vectorstore_faiss,query, PROMPT):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 10}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']

prompt_template_compliance_upd = """
Imagine you are a compliance officer for a bank checking if policies and guidelines are being met.
Check the sections of the following question on whether the policies and guidelines are being met.
<question>
{question}
</question

These are the sections of the above document to be checked with the below policies and guidelines.
<sections>
{section}
</sections

The following are the policies and guidelines to be checked against:
<context>
{context}
</context

Provide a high level response table of the question with a single word - Yes, No or Partially Compliant for each section of the question as a table.
The first column of the table is the section, second column is the check for compliance with a Yes, No or Partially Compliant.

Provide a detailed summary called as "SUMMARY SECTION" under the high level response table for the non compliant or partially compliant 
sections as a key value pair with quoted reference from the context above and suggested change. 
The key being the section name of the non compliant or partially compliant section 
and the value being a standardised reponse with detalied summary called as "Detailed Summary" and suggested change called as "Suggested Change" 
as two separate paragraphs. 

Please refer only to the document. 
Please be formal in your response. 
Please avoid any biases.
Assistant:"""



# sections_txt = '''
# 1. Introduction
# 2. Scope
# 3. Governance and Strategy
# 4. Risk Management Framework
# 5. Information Security
# 6. ICT Operations Management
# 7. Business Continuity Management
# 8. Compliance and Reporting
# 9. Training and Awareness
# '''

# PROMPT = PromptTemplate(
#     template=prompt_template_compliance_upd.replace('{section}',sections_txt), input_variables=["context", "question"]
# )

def get_claude_llm():
    ##create the Anthropic Model
    llm=BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0",client=bedrock,
                model_kwargs={'max_tokens':1000, 'temperature':0})
    
    return llm

# Function to display compliance reasons
def compliance_reasons(section, reasons, status, status_colors):

    if section in reasons:
        # print('test')

        st.write('**Reasons:**')
        # st.write(reasons[section][1])
        st.markdown(f"<div class='status' style='{status_colors[status.title()]}'>{reasons[section][0].title()}</div>", unsafe_allow_html=True)
        st.write('**Suggested Change:**')

        st.markdown(f"<div class='status' style='background-color: lightgrey; color: black;'>{reasons[section][1].title()}</div>", unsafe_allow_html=True)
        st.write("")
    else:
        st.write('')

# Function to parse the high-level response table
def parse_high_level_response(text):
    pattern = re.compile(r"\| (.+?) \| (.+?) \|")
    matches = pattern.findall(text)
    data = [match for match in matches if match[0] != 'Section']
    df = pd.DataFrame(data, columns=['Section', 'Compliance'])
    df = df.loc[df['Compliance'].str.lower()!='compliance']
    return df

# Function to parse the detailed summary
# def parse_detailed_summary(text):
#     sections = re.split(r"(\d+\.\s.+?:\n)", text)[1:]
#     data = []
#     for i in range(0, len(sections), 2):
#         section = sections[i].strip().replace('\n', '')
#         details = sections[i+1].strip()
#         compliance, details = details.split(' - ', 1)
#         suggested_change = re.search(r"Suggested change: (.+)", details).group(1)
#         data.append([section, compliance, details, suggested_change])
#     df = pd.DataFrame(data, columns=['Section', 'Compliance', 'Details', 'Suggested Change'])
#     df['Section'] = df['Section'].str.replace(':', '', regex=False)
#     df['Details']=df['Details'].apply(lambda x: x.split('\n')[0])
#     return df

def parse_detailed_summary(text):
    sections = re.split(r"(\d+\.\s.+?:\n)", text)[1:]
    data = []
    for i in range(0, len(sections),2):
        section = sections[i].strip().replace('\n', '')
        summary = sections[i+1].strip()
        detail_s, suggested_c = summary.split("suggested change")
        detail_s = detail_s.replace('"detailed summary":', '')
        suggested_c = suggested_c.replace('":', '')
        data.append([section, detail_s, suggested_c])
    df = pd.DataFrame(data, columns=['Section', 'Detailed Summary', 'Suggested Change'])
    df['Section'] = df['Section'].str.replace(':', '', regex=False)
    df["Section"] = df["Section"].str.replace('"','', regex=False)
    return df

## Chain to pull sections of a document
chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Give me the sections of the following Document. Call it 'Sections:' and provide it as a numbered list:\n\n{doc}")
    | get_claude_llm()
    | StrOutputParser()
)

def display():
    st.write("CompliCheck is your go-to solution for all compliance checks. Simply upload your documents, and our advanced AI will analyze and ensure that all regulatory requirements are met. Stay compliant with ease and confidence.")

    # Example upload widget
    uploaded_file = st.file_uploader("✍️ Upload your compliance document")
    user_question = None
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        user_question = extract_pdf(uploaded_file)   

    option = st.selectbox(
    "🏆 Choose the guidelines to check against",
    # ("FINRA", "BIA", "Regulatory", "External Audit", "Legal and Compliance")
    ("EBA",'Anti-Bribery', "Fraud", "Cyber Security", "FINRA", "SEC"),
    index=None,
    placeholder="Select Guideline"
    )

    filenames = {
        'EBA' : 'Final draft Guidelines on ICT and security risk management.pdf',
        'Fraud' : 'Bank_Payment_Fraud_Awareness_Guide.pdf',
        'Cyber Security' : 'Bank_Designing_For_CyberSecurity.pdf',
        'Anti-Bribery' : 'Bank_Designing_For_CyberSecurity.pdf',
    }

    if option == "EBA"  and user_question is not None:
        try: 
            with st.spinner("Processing... uploaded to vector Database"):  
                f"faiss_index_{inp}"     
                faiss_index = FAISS.load_local(file_path/"VectorDB"/f"faiss_index_{option.replace(' ','')}", bedrock_embeddings, allow_dangerous_deserialization=True)
        except:
            with st.spinner("Creating Vector DB..."):
                docs = data_ingestion(filenames[option])
                save_vector_store(docs, option.replace(' ',''))
                st.success("Done")
                faiss_index = FAISS.load_local(file_path/"VectorDB"/f"faiss_index_{option.replace(' ','')}", bedrock_embeddings, allow_dangerous_deserialization=True)
                st.success("Updated FAISS Index Populated") 
        
        llm=get_claude_llm() 

        with st.spinner("Processing... Invoking LLM to pull sections for the context and Prompt") :

            docs_inp = Document(page_content=user_question)
            summaries = chain.batch([docs_inp], {"max_concurrency": 5})
            sections_txt = summaries[0].split('Sections:\n')[-1]
            PROMPT = PromptTemplate(
                template=prompt_template_compliance_upd.replace('{section}',sections_txt), input_variables=["context", "question"]
            )
        
        with st.spinner("Processing... LLM to check conformance"):
            llm_response =  get_response_llm(llm,faiss_index,user_question, PROMPT)  
        # st.write(get_response_llm(llm,faiss_index,user_question, PROMPT))
        st.success("Done")

    if option == "Anti-Bribery"  and user_question is not None:
        try: 
            with st.spinner("Processing... uploaded to vector Database"):       
                faiss_index = FAISS.load_local(f"VectorDB\\faiss_index_{option.replace(' ','')}", bedrock_embeddings, allow_dangerous_deserialization=True)
        except:
            with st.spinner("Creating Vector DB..."):
                docs = data_ingestion(filenames[option])
                save_vector_store(docs, option.replace(' ',''))
                st.success("Done")
                faiss_index = FAISS.load_local(f"VectorDB\\faiss_index_{option.replace(' ','')}", bedrock_embeddings, allow_dangerous_deserialization=True)
                st.success("Updated FAISS Index Populated") 
        
        llm=get_claude_llm() 

        with st.spinner("Processing... Invoking LLM to pull sections for the context and Prompt") :

            docs_inp = Document(page_content=user_question)
            summaries = chain.batch([docs_inp], {"max_concurrency": 5})
            sections_txt = summaries[0].split('Sections:\n')[-1]
            PROMPT = PromptTemplate(
                template=prompt_template_compliance_upd.replace('{section}',sections_txt), input_variables=["context", "question"]
            )
        
        with st.spinner("Processing... LLM to check conformance"):
            llm_response =  get_response_llm(llm,faiss_index,user_question, PROMPT)  
        # st.write(get_response_llm(llm,faiss_index,user_question, PROMPT))
        st.success("Done")

    if option == "Fraud" :
        with st.spinner("Processing..."):  
            try:  
                faiss_index = FAISS.load_local(f"VectorDB\\faiss_index_{option.replace(' ','')}", bedrock_embeddings, allow_dangerous_deserialization=True)
                st.success("FAISS Index Populated") 
            except:
                with st.spinner("Creating Vector DB..."):
                    docs = data_ingestion(filenames[option])
                    save_vector_store(docs, option.replace(' ',''))
                    st.success("Done")
                    faiss_index = FAISS.load_local(f"VectorDB\\faiss_index_{option.replace(' ','')}", bedrock_embeddings, allow_dangerous_deserialization=True)
                    st.success("FAISS Index Populated") 

    if option == "Cyber Security" :
        with st.spinner("Processing..."):  
            try:          
                faiss_index = FAISS.load_local(f"VectorDB\\faiss_index_{option.replace(' ','')}", bedrock_embeddings, allow_dangerous_deserialization=True)
                st.success("FAISS Index Populated") 
            except:
                
                with st.spinner("Creating Vector DB..."):
                    docs = data_ingestion(filenames[option])
                    save_vector_store(docs, option.replace(' ',''))
                    st.success("Done")
                    faiss_index = FAISS.load_local(f"VectorDB\\faiss_index_{option.replace(' ','')}", bedrock_embeddings, allow_dangerous_deserialization=True)
                    st.success("FAISS Index Populated") 
   
    else:
        pass
    
    # Extracting the relevant parts of the text
    if option is not None:
        # print(llm_response)


        split_text = re.split(r"(summary section:)", llm_response.lower(), flags=re.IGNORECASE)
        high_level_text = split_text[0]
        detailed_summary_text = split_text[-1]


        # Creating the dataframes
        high_level_df = parse_high_level_response(high_level_text)
        detailed_summary_df = parse_detailed_summary(detailed_summary_text)

        # Create two columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='header table'>My Doc</div>", unsafe_allow_html=True)
            with st.container():
                for section in high_level_df['Section'].to_list():
                    with st.expander(section.title()):

                        # st.write(f"{section} details...")
                        st.write("**Accept CompliEase change for section?**")
                        st.radio(f"", ["Yes", "No"], index=0, key=f"{section}_choice")

        with col2:

            # result table 
            st.markdown("<div class='header-compliance table'>Am I compliant?</div>", unsafe_allow_html=True)
            compliance_status = high_level_df.set_index('Section').T.to_dict('records')[0]

            status_colors = {
                "Yes": "background-color: lightgreen; color: black;",
                "No": "background-color: lightcoral; color: black;",
                "Partially Compliant": "background-color: lightyellow; color: black;"
            }


            with st.container():
                reasons = detailed_summary_df.set_index('Section').T.to_dict('list')
                for section, status in compliance_status.items():
                    with st.expander(f"{status.title()}"):
                        st.markdown(f"<div class='status' style='{status_colors[status.title()]}'>{section.title()}: {status}</div>", unsafe_allow_html=True)
                        compliance_reasons(section, reasons, status, status_colors)
    
    else:
        pass