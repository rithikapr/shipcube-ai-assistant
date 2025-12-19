from utils.ai_model import get_retrieval_answer


def vectorstore_retrieval(refined_query: str, top_k: int =3, threshold: float =0.75):
    """
    Retrieves top FAQ matches for the given query.
    """
    faq_context = "Query: " + refined_query.get('query') + " || Context: " + refined_query.get('context')
    faqs = get_retrieval_answer(faq_context, top_k=top_k, score_threshold=threshold)

    retrieved_chunks = ""
    metadata = ""

    for item, score in faqs:
        retrieved_chunks += f"Data_Chunk: {item['answer']} || Score: {score}\n "
        metadata += f"Tags: {item['tag']} || Score: {score}\n "
    
    return retrieved_chunks, metadata 



