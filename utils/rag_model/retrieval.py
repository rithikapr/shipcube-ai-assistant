from utils.ai_model import get_retrieval_answer


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
    """
    Retrieves top FAQ matches for the given query.
    """
    faq_context = "Query: " + refined_query.get('query') + ". Context: " + refined_query.get('context')
    print(f"[Retrieval] Faq Context: {faq_context}")
    faqs = get_retrieval_answer(faq_context, top_k=top_k, score_threshold=threshold)

    retrieved_chunks = ""
    metadata = ""

    for item in faqs:
        retrieved_chunks += f"Data_Chunk: {item['answer']} || Score: {item['score']}\n "

    return retrieved_chunks, metadata


