from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from utils.rag_model.llm import llm


router_template = """
    You are the primary router for the ShipCube AI assistant.
    Classify the User Input into exactly ONE of the following categories.

    Return ONLY valid JSON. Do NOT include explanations or extra text.

    1. "small_talk":
    Greetings, compliments, casual conversation.
    OUTPUT:
    {{"type": "small_talk"}}

    2. "pricing":
    Questions about costs, rates, fees, invoices, billing.
    OUTPUT:
    {{"type": "pricing"}}

    3. "order_tracking":
    Questions about locating orders, shipment status, tracking IDs, delivery.
    OUTPUT:
    {{"type": "order_tracking"}}

    4. "technical":
    Questions about ShipCube services, warehousing, logistics, facilities.
    OUTPUT:
    {{"type": "technical"}}

    User Input: {query}

    Return JSON ONLY.
    """

router_chain = (
    PromptTemplate.from_template(router_template) 
    | llm 
    | JsonOutputParser()
)


refine_template = """
    Given a chat history and a follow-up question, summarise the history so that most important information retain, in the format you understand context.
    Send the rephrased question along with the super summarized form of the chat history. 
    If the history is empty or irrelevant, return the question as is and context as empty string.
    
    Chat History:
    {chat_history}
    
    Follow Up Input: {question}
    
    Output: 
    Return VALID JSON ONLY in the following format: 
    {{
        "query": string,
        "context": string
    }}

"""

refine_chain = (
    PromptTemplate.from_template(refine_template) 
    | llm 
    | JsonOutputParser()
)


rag_answer_template = """
    You are a Retrieval-Augmented Generation (RAG) answer synthesis engine.

    Your task is to answer the user question **using ONLY the provided retrieved context**.
    If the answer is not present in the context, explicitly say so.

    ### Rules
    - Use ONLY the information in the retrieved context.
    - Do NOT hallucinate or use external knowledge.
    - Preserve technical accuracy, numbers, constraints, and terminology.
    - Be concise and precise.
    - If multiple context chunks conflict, state the uncertainty.
    - If no answer is found, return `"answer": null`.
    - Retrieved relevant context chunks are marked as Context and Score. The higher the score, the more relevant the chunk. Not everything in the context may be useful.

    ### Inputs

    User Question:
    {question}

    History Context:
    {context}

    Retrieved Context Chunks:
    {retrieved_chunks}

    Additional Metadata (optional):
    {metadata}

    ### Output
    Return **VALID JSON ONLY** in the following format:

    {{
        "answer": string | null, 
        "confidence": float, 
        "sources": [ {{ "chunk_id": string, "relevance": float }} ], 
        "notes": string | null 
    }}

    ### Field Guidelines
    - answer: Direct answer to the question, or null if not answerable.
    - confidence: A number between 0.0 and 1.0 representing confidence.
    - sources: Only include chunks actually used in the answer.
    - relevance: How strongly that chunk supports the answer (0.0–1.0).
    - notes: Any ambiguity, assumptions, or missing information.
    """

rag_answer_chain = (
    PromptTemplate.from_template(rag_answer_template)
    | llm
    | JsonOutputParser()
)


small_talk_template = """
    You are a friendly conversational assistant for ShipCube.

    Your role is to respond naturally to casual or social user messages
    such as greetings, thanks, jokes, curiosity, or light conversation.

    Guidelines:
    - Be warm, polite, and engaging.
    - Use a relaxed and friendly tone.
    - Keep responses short and conversational.
    - Do NOT mention databases, AI models, or internal systems.
    - Do NOT ask technical questions.
    - Creativity and personality are encouraged.
    - High temperature behavior is expected.

    User Message:
    {user_query}

    Context:
    {context}

    Output: 
    Return VALID JSON ONLY in the following format: 
    {{
        "answer": string 
    }}
"""

small_talk_chain = (
    PromptTemplate.from_template(small_talk_template)
    | llm 
    | JsonOutputParser()
)