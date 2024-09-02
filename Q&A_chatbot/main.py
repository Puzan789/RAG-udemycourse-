from langchain_google_genai import ChatGoogleGenerativeAI ,GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.prompts.chat import (
    HumanMessagePromptTemplate , SystemMessagePromptTemplate
)
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from dotenv import load_dotenv 
load_dotenv()
import os 

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

template:str ="""
You are a customer support specialist  /
question: {question} .You assist users with the inquiries  based on {context} / and tecnical issues.
"""
system_message_prompt=SystemMessagePromptTemplate.from_template(template)
human_message_prompt=HumanMessagePromptTemplate.from_template(input_variables=["question","context"],
                                                              template="question")
chat_prompt_template=ChatPromptTemplate.from_messages(
    [system_message_prompt,human_message_prompt]
)
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro') 


def load_documents():
    loader=TextLoader("./man.txt")
    documents=loader.load()
    texts=CharacterTextSplitter(chunk_size=100,chunk_overlap=10)
    return texts.split_documents(documents)
      

def load_embeddings(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db=Chroma.from_documents(documents,embeddings)
    return db.as_retriever()
     

def generate_response(retriever,query):
    chain=( 
        {"context":retriever,"question":RunnablePassthrough()}|chat_prompt_template | model | StrOutputParser
     )   
     
    return chain.invoke(query)

 
def query(query):
    documents=load_documents()
    retriever=load_embeddings(documents)
    response=generate_response(retriever,query)
    return response



def start():
    print("Type your questions and press Enter\n")
    print("===")
    choice = input("Enter your choice: ")
    if choice == "1":
        ask()
    elif choice == "2":
        print("Goodbye!")
        exit()
    else:
        print("Invalid choice!")
        start()
    
def ask():
    while True:
        question = input("Ask a question: ")
        response = query(question)
        print(f"Answer: {response}")
        choice = input("Do you want to ask another question? (yes/no): ")
        if choice.lower() != "yes":
            print("Goodbye!")
            break

if __name__ == "__main__":
    start()
