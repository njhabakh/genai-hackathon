from typing import Type
import boto3
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import streamlit as st
import pandas as pd

## We will be suing Titan Embeddings Model To generate Embedding
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.chat_models.bedrock import BedrockChat

## Data Ingestion
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector Embedding And Vector Store
from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain_experimental.graph_transformers import (
    LLMGraphTransformer,
)

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langchain_core.documents import Document

st_callback = StreamlitCallbackHandler(st.container())

## Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-image-v1", client=bedrock
)


## Data ingestion
def loader():
    loader = PyPDFDirectoryLoader("pdfs")
    documents = loader.load()
    docs = [doc for doc in documents if doc.page_content.strip()]
    return docs


def data_ingestion(documents):
    # - in our testing Character split works better with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

    docs = text_splitter.split_documents(documents)
    return docs


## Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index_pdfs")


def get_llama2_llm():
    ##create the Llama Model
    llm=Bedrock(model_id="meta.llama2-70b-chat-v1",client=bedrock,
                model_kwargs={'max_gen_len':200})
    
    return llm


def get_mistral_llm():
    ##create the Mistral Model
    llm = Bedrock(
        model_id="mistral.mistral-7b-instruct-v0:2",
        client=bedrock,
    )
    llm.model_kwargs = {
        "temperature": 0.3,
        "max_tokens": 1000,
    }
    return llm

def get_claude_llm():
    ##create the Anthropic Model
    llm=BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0",client=bedrock,
                model_kwargs={'max_tokens':200})

def get_llm_transformer(llm):
    transformer = LLMGraphTransformer(llm=llm)
    return transformer



def get_current_stock_return(ticker):
    df = pd.read_csv(f"data/{ticker}.csv", index_col = 0)
    return (((1 + df["Adj Close"].ffill().pct_change()).cumprod().iloc[-1]) - 1)*100


class CurrentStockPriceInput(BaseModel):
    """Inputs for get_current_stock_price"""

    ticker: str = Field(description="Ticker symbol of the stock")


class CurrentStockReturnTool(BaseTool):
    name = "get_current_stock_return"
    description = "Useful when you want to get current stock return"
    args_schema: Type[BaseModel] = CurrentStockPriceInput  # type: ignore

    def _run(self, ticker):
        return get_current_stock_return(ticker)

    def _arun(self, ticker):
        raise NotImplementedError("func get_current_stock_return did not support async.")


prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use atleast summarize with 
100 words with detailed explantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.

{agent_scratchpad}

<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    answer = qa({"query": query})
    return answer["result"]


def configure_retriever():
    bedrock_embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-image-v1", client=bedrock
    )
    vectorstore = FAISS.load_local(
        "faiss_index_pdfs", bedrock_embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore


latest_stock_return = CurrentStockReturnTool(
    name="get_current_stock_return",
    description="Get the latest stock return for a given ticker symbol.",
)

# print("init retriever tool")


def search_docs(query):
    """Searches the document store for relevant information."""
    vectorstore = configure_retriever()
    print(f"query: {query}")
    if query["value"]:
        results = vectorstore.similarity_search(query["value"])
    else:
        results = vectorstore.similarity_search(query["query"])
    # results = vectorstore.similarity_search(query)
    return {"docs": results}


retriever_tool = Tool(
    name="search_docs",
    func=search_docs,
    description="Search the document store for relevant information.",
)

# print("init tools")
tools = [retriever_tool]
print("init openai functions")
# llm = get_llama2_llm()
llm = get_mistral_llm()
# llm = get_claude_llm()
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=ConversationBufferMemory(),
    handle_parsing_error = True,
    verbose=True,
)

def main():
    st.header("Agent")

    user_question = st.text_input("Ask a Question")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion(loader())
                get_vector_store(docs)
                st.success("Done")

    if st.button("Run Agent"):
        with st.spinner("Processing..."):
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent.invoke(
                {"input": user_question}, {"callbacks": [st_callback]}
            )
            st.write(response["output"])
            st.success("Done")


if __name__ == "__main__":
    main()