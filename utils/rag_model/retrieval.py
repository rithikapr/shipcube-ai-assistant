from utils.ai_model import get_retrieval_answer
from utils.rag_model.prompts import (
    router_chain,
    refine_chain,
    rag_answer_chain,
    small_talk_chain
)


"""
    *   @brief Takes the complete query information and chat history to retrieve relevant FAQ matches from the vector store.

    *   @param refined_query: A string containing the user's refined query and context.
    *   @param top_k: An integer specifying the number of top matches to retrieve. Default is 3.
    *   @param threshold: A float specifying the minimum score threshold for retrieval. Default is 0.33.
    
    *   @pre  refined_query must be a valid string containing user query and context.
    *   @pre  top_k must be a positive integer.
    *   @pre  threshold must be a float between 0.0 and 1.0.
    
    *   @return retrieved_chunks: A string containing the retrieved FAQ chunks formatted for further processing.
    *   @return metadata: A string containing any additional metadata related to the retrieval process.

"""
def vectorstore_retrieval(refined_query: str, top_k: int =3, threshold: float =0.33):
    faq_context = "Query: " + refined_query.get('query') + ". Context: " + refined_query.get('context')
    print(f"[Retrieval] Faq Context: {faq_context}")
    faqs = get_retrieval_answer(faq_context, top_k=top_k, score_threshold=threshold)

    retrieved_chunks = ""
    metadata = ""

    for item in faqs:
        retrieved_chunks += f"Data_Chunk: {item['answer']} || Score: {item['score']}\n "

    return retrieved_chunks, metadata



"""
    *   @brief Determines the user intent from the given query using a language model.

    *   @param query: A string containing the user's query.
    
    *   @pre  query must be a valid string containing user query.
    
    *   @return intent: A string representing the determined user intent.

"""
def get_user_intent(query: str):
    route = router_chain.invoke({"query": query})
    intent = route.get("type")

    return intent



"""
    *   @brief Transforms the user query based on chat history to refine it for better retrieval.

    *   @param user_query: A string containing the user's query.
    *   @param chat_history: A string containing the chat history.

    *   @pre  user_query must be a valid string containing user query.
    *   @pre  chat_history must be a valid string containing chat history.

    *   @return refined query: A Dictionary in the following format:
                {
                    "query": string,
                    "context": string
                }

"""
def get_refined_query(chat_history: str, user_query: str):
    refined_query = refine_chain.invoke({
        "chat_history": chat_history,
        "question": user_query
    })
            
    return refined_query



"""
    *   @brief Generates an answer to the user's question using retrieved context chunks and metadata.

    *   @param refined_query: A Dictionary containing the refined user query and context.
    *   @param retrieved_chunks: A string containing the retrieved context chunks.
    *   @param metadata: A string containing any additional metadata.

    *   @pre  question must be a valid string containing user question.
    *   @pre  context must be a valid string containing context.
    *   @pre  retrieved_chunks must be a valid string containing retrieved context chunks.
    *   @pre  metadata must be a valid string containing additional metadata.

    *   @return rag_response: A Dictionary in the following format:
                {
                    "answer": string, 
                    "confidence": float, 
                    "notes": string
                }

"""
def get_rag_answer(refined_query, retrieved_chunks: str, metadata: str):
    question = refined_query.get("query")
    context = refined_query.get("context")

    rag_response = rag_answer_chain.invoke({
        "question": question,
        "context": context,
        "retrieved_chunks": retrieved_chunks,
        "metadata": metadata
    })

    return rag_response



"""
    *   @brief Generates an answer to the user's question using general small talk context and intelligence.

    *   @param refined_query: A Dictionary containing the refined user query and context.

    *   @pre  refined_query must be a valid Dictionary containing user query and context.

    *   @return response: string containing the small talk answer.

"""
def small_talk_response(refined_query):
    response = small_talk_chain.invoke({"user_query": refined_query.get('query'), "context": refined_query.get("context")})

    return response.get('answer')


