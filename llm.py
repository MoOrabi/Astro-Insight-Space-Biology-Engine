import os
from dotenv import load_dotenv
from groq import Groq

from test_query import get_question_answer as retrieve_relevant_chunks

# Load environment variables from the .env file

load_dotenv()
# Securely get the API key from the environment

api_key = os.getenv("GROQ_API_KEY")
# Check if the API key is available

if not api_key:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")
# Initialise the Groq client

llm_client = Groq(api_key=api_key)

def generate_llm_response(user_prompt: str, relevant_chunks: list[dict]) -> dict:
    
    #context = "\n\n---\n\n".join(relevant_chunks)
    
    context_documents = [chunk["document"] for chunk in relevant_chunks]
    context = "\n\n---\n\n".join(context_documents)
    
    citations = [chunk["metadata"] for chunk in relevant_chunks]
    
    system_prompt = (
        "You are an expert AI assistant for the 'Astro-Insight Space Biology Engine'. "
        "Your purpose is to help scientists answer questions about space biology "
        "using ONLY the provided scientific literature."
        "\n\nInstructions:"
        "\n1. Act as a specialist in space biology."
        "\n2. You must answer the user's question **based ONLY on the information provided in the 'CONTEXT' section**."
        "\n3. Do not use any external knowledge from your own training data."
        "\n4. If the provided context does not contain enough information, you must explicitly state: "
        "'Based on the provided documents, I cannot answer this question.'"
    )
    
    final_prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"---\n\n"
        f"Based on the context above, please answer the following question:\n"
        f"Question: {user_prompt}"
    )
    
    try:
        chat_completion = llm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt},
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=2048,
            top_p=1,
            stop=None,
            )
        llm_text = chat_completion.choices[0].message.content
        #return chat_completion.choices[0].message.content
        return {"answer": llm_text, "citations": citations}
    
    except Exception as e:
        '''
        print(f"An error occurred while calling the Groq API: {e}")
        return "Sorry, I encountered an error while generating a response."
        '''
        print(f"An error occurred while calling the Groq API: {e}")
        # Return a structured error response that matches the expected format.
        return {"answer": "Sorry, I encountered an error while generating a response.", "citations": []}
    
def get_cited_answer(user_question: str) -> dict:
    """
    The main RAG pipeline function. It orchestrates the retrieval and generation process.
    """
    print("Step 1: Retrieving relevant documents from the knowledge base...")
    # Call the imported function to get chunks and metadata from ChromaDB
    relevant_chunks = retrieve_relevant_chunks(user_question)
    
    if not relevant_chunks:
        return {"answer": "I could not find any relevant information in the knowledge base to answer your question.", "citations": []}

    print("Step 2: Generating a synthesised answer with citations...")
    # Pass the retrieved data to the generator
    response = generate_llm_response(user_question, relevant_chunks)
    
    return response

# 5. An executable block to run a live demonstration
if __name__ == '__main__':
    # Example usage:
    question = "What was the purpose of the Bion-M 1 mission and what was the condition of the mice after the flight?"
    
    print(f"--- Querying the Astro-Insight Engine ---")
    print(f"Question: {question}\n")
    
    final_response = get_cited_answer(question)
    
    print("\n--- AI-Generated Answer ---")
    print(final_response['answer'])
    print("\n--- Citations ---")
    if final_response['citations']:
        for i, citation in enumerate(final_response['citations'], 1):
            print(f"[{i}] {citation['title']} ({citation['year']})")
            print(f"    URL: {citation['url']}")
    else:
        print("No citations found.")
