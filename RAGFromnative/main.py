from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
import warnings
from langchain.chains import RetrievalQA
warnings.filterwarnings("ignore")

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Loading the model
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

# Loading the document
loader = WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")
data = loader.load()

# Defining prompt templates
template = "You are a developer who answers {question}based on the knowledge provided on the {context}. you say i am not train for it if the question is asked outside the {context}"
prompt = PromptTemplate(template=template, input_variables=["question", "context"])

# Retriever 
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
retriever = FAISS.from_documents(data, embeddings)

# Defining the function to generate the response 
def generatequery(query):
    chain_kwargs = {"prompt": prompt}
    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever.as_retriever(search_kwargs={"k": 1}),
        chain_type_kwargs=chain_kwargs 
    )
    response = chain.run(query)
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
        response = generatequery(question)
        print(f"Answer: {response}")
        choice = input("Do you want to ask another question? (yes/no): ")
        if choice.lower() != "yes":
            print("Goodbye!")
            break

if __name__ == "__main__":
    start()
