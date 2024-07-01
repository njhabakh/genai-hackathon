# **Compliance Application** 

This project creates an application that uses [RAG (Retrieval-Augmented Generation)](https://aws.amazon.com/what-is/retrieval-augmented-generation/) for document contextualization. The application has the following features:
- CompliCheck : Analyze and ensure all requirements and guidelines are met be it regulatory, audit or compliance
- ReguSync : Compare different versions of your documents to see what's changed
- CompliBot : Q&A chat bot with your requirements and guidelines



![Preview](conformant.gif)

## **Features** 

- **Streamlit**: For a smooth web application interface.
- **Langchain**: Integrated for advanced functionalities.
- **AWS Services**: Harness the power of Amazon's cloud services.
    - **Amazon Bedrock**: A fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Stability AI, and Amazon with a single API, along with a broad set of capabilities you need to build generative AI applications, simplifying development while maintaining privacy and security

## **Getting Started** 

### **1. Pre-requisites** 
1. Have the following software on your system:
    [git](https://git-scm.com/download)
    [IDE](https://code.visualstudio.com/download)
    [python](https://www.python.org/downloads/)

2. Clone the repository to your local machine (preferably inside a git or code folder).
    ```bash
    git clone https://github.com/njhabakh/genai-hackathon.git
    ```

### **2. Set Up** 

1. Create the Python virtual env `venv` inside the cloned folder :
    ```bash
    python -m venv venv
    ```

2. Activate the python environment:
    ```bash
    venv/bin/activate.csh
    ```
With your virtual environment active, install the necessary packages:

3. Install all the necessary packages:
    ```bash
    pip install -r requirements.txt
    ```
This command installs all dependencies from the `requirements.txt` file into your `data-search-env` environment.

4. Configure aws settings:
Look at this [video](https://www.youtube.com/watch?v=2maPaQutcWs&t=95s) for a step by step guide on setting up your Bedrock account and access keys.
    ```bash
    aws configure 
    AWS Access Key ID [None] : <Enter AWS Access key>
    AWS Secret Access Key [None] : <Enter AWS Secret Access key>
    Default region name [None]: us-east-1
    Default output format [None]: json
    ```

### **3. Usage**

To launch the application:
1. Launch the application using Streamlit:
   ```bash
   streamlit run app.py 
   ```

2. Your default web browser will open, showcasing the application interface.

3. Follow the on-screen instructions to load your data and start using conformant.
