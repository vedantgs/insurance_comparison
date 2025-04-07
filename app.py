import streamlit as st
from typing import List
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()

# Streamlit UI
st.title("Insurance Plan Comparator")

# Retrieve API key from environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")
print("OpenAI API Key:", openai_api_key)  # Debugging line

uploaded_file_a = st.file_uploader("Upload Insurance Plan A (PDF)", type=["pdf"])
uploaded_file_b = st.file_uploader("Upload Insurance Plan B (PDF)", type=["pdf"])

if uploaded_file_a and uploaded_file_b and openai_api_key:
    with open("plan_a.pdf", "wb") as f:
        f.write(uploaded_file_a.getbuffer())
    
    with open("plan_b.pdf", "wb") as f:
        f.write(uploaded_file_b.getbuffer())
    
    # Load documents
    loader_a = UnstructuredLoader(
        file_path="plan_a.pdf",
        chunking_strategy="by_title",
        max_characters=3000,
        new_after_n_chars=2000,
        overlap=500
    )
    loader_b = UnstructuredLoader(
        file_path="plan_b.pdf",
        chunking_strategy="by_title",
        max_characters=3000,
        new_after_n_chars=2000,
        overlap=500
    )

    docs_a = loader_a.load()
    docs_b = loader_b.load()

    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small", openai_api_key=openai_api_key)

    for doc in docs_a:
        for md in doc.metadata:
            doc.metadata[md] = str(doc.metadata[md])

    for doc in docs_b:
        for md in doc.metadata:
            doc.metadata[md] = str(doc.metadata[md])

    # Store in FAISS
    faiss_db_a = FAISS.from_documents(docs_a, embeddings)
    faiss_db_b = FAISS.from_documents(docs_b, embeddings)

    # Define response model
    class InsurancePlanInformation(BaseModel):
        answer_for_plan_a: str = Field(description="The response to the user query for plan A")
        answer_for_plan_b: str = Field(description="The response to the user query for plan B")
        comparative_results: str = Field(description="Comparison between plan A and plan B")

    llm = ChatOpenAI(model_name="gpt-4o-mini", max_tokens=2500, openai_api_key=openai_api_key)
    system_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant tasked with comparing two insurance plans based on retrieved information. Provide detailed responses for each plan and a comparative analysis for a given user question. Respond to the query in a detailed format. Do not hallucinate information in case there is no data relevant to the user query."),
        ("human", "Here is the information for Plan A:\n{plan_a_info}\n\nAnd here is the information for Plan B:\n{plan_b_info}\n\n Answer the question: {user_query}.")
    ])

    test = system_prompt | llm.with_structured_output(InsurancePlanInformation)

    def compare_insurance_plans(user_query: str, top_k: int = 3) -> InsurancePlanInformation:
        """
        Retrieve relevant documents for each insurance plan, format the retrieved content, and invoke the language model to compare the plans based on the user's query.
        """
        relevant_docs_a = faiss_db_a.similarity_search(user_query, k=top_k)
        relevant_docs_b = faiss_db_b.similarity_search(user_query, k=top_k)

        plan_a_info = "\n".join([doc.page_content for doc in relevant_docs_a])
        plan_b_info = "\n".join([doc.page_content for doc in relevant_docs_b])

        response = test.invoke({
            "plan_a_info": plan_a_info,
            "plan_b_info": plan_b_info,
            "user_query": user_query
        })

        return response

    user_query = st.text_input("Enter your question about the insurance plans:")

    if st.button("Compare Plans"):
        if user_query:
            try:
                result = compare_insurance_plans(user_query)

                st.subheader("Plan A Response:")
                st.write(result.answer_for_plan_a)

                st.subheader("Plan B Response:")
                st.write(result.answer_for_plan_b)

                st.subheader("Comparative Analysis:")
                st.write(result.comparative_results)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please enter a question.")
else:
    if not openai_api_key and (uploaded_file_a or uploaded_file_b):
        st.warning("Please set your OpenAI API key in the environment variables to proceed.")
    elif not uploaded_file_a or not uploaded_file_b:
        st.info("Please upload both insurance plans to compare them.")
