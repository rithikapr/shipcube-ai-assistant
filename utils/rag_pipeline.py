from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from utils.ai_model import generate_answer_from_retrieval
import dotenv
import json
import re
from flask import g
import sqlite3
from utils.ai_model import (
    classify_order_intent,
    BASIC_RESPONSE_PROMPT,
    DETAILED_RESPONSE_PROMPT
)


dotenv.load_dotenv()

ORDER_ID_RE = re.compile(
    r"(order|tracking)\s*(id|number)?\s*[:#]?\s*(\d{9})(?:\.0)?",
    re.I
)

BASIC_FIELDS = {
    "order_number",
    "order_date",
    "carrier",
    "shipping_method",
    "tracking_number",
}

DETAILED_FIELDS = {
    "to_name",
    "zip",
    "state",
    "country",
    "warehouse",
    "tpl_customer",
    "size_dimensions",
    "weight_oz",
    "final_amount",
}


def apply_data_policy(order: dict, intent: str) -> dict:
    allowed = (
        BASIC_FIELDS | DETAILED_FIELDS
        if intent == "DETAILED_ORDER_INFO"
        else BASIC_FIELDS
    )
    return {k: v for k, v in order.items() if k in allowed}


def get_db():
    return sqlite3.connect("data/shipcube.db")

def normalize_order_token(token: str):
    """
    Handles Excel numeric artifacts like 696381280.0
    """
    token = token.strip()
    variants = {token}

    # if pure digits → add .0 variant
    if token.isdigit():
        variants.add(f"{token}.0")

    # if ends with .0 → add integer variant
    if token.endswith(".0"):
        variants.add(token[:-2])

    return list(variants)
    
def find_order_in_db(order_token: str):
    conn = sqlite3.connect("data/shipcube.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    token = order_token.strip()
    variants = {token}

    if token.isdigit():
        variants.add(f"{token}.0")
    if token.endswith(".0"):
        variants.add(token[:-2])

    variants = list(variants)
    normalized_variants = []
    for v in variants:
        try:
            normalized_variants.append(int(float(v)))
        except Exception:
            normalized_variants.append(v)

    cur.execute(f"""
    SELECT *
    FROM client_orders
    WHERE CAST(order_id AS INTEGER) IN ({','.join('?' * len(normalized_variants))})
       OR order_number IN ({','.join('?' * len(normalized_variants))})
       OR tracking_number IN ({','.join('?' * len(normalized_variants))})
    LIMIT 1
""", normalized_variants * 3)


    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

"""
*   @brief Create a RAG Agent that routes queries, refines them, and retrieves answers.

*   @details The RAGAgent class encapsulates the logic for handling user queries by:
    1. Routing the query to determine if it's small talk, pricing-related, or technical.
    2. Refining the query to make it standalone and specific.
    3. Retrieving an answer from a knowledge base using a RAG approach.
    4. Formatting the final response with the answer, source, and original query.

"""
class RAGAgent:

    """
    *   @brief Initializes the RAGAgent with LLM, routing, and refinement chains.

    *   @param Nothing
    
    *   @pre  Global chat model is ready with a temporary / logged in user.

    *   @return Nothing
 
    *   @post  Guarantees that the llm is initialized with a ready-to-use chat model
    *          and that the routing and refinement chains are set up for processing queries.
    
    """
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0
        )
        
        router_template = """
            You are the primary router for the ShipCube AI assistant.
            Classify the User Input into exactly ONE of the following categories.

            Return ONLY valid JSON. Do NOT include explanations or extra text.

            1. "small_talk":
            Greetings, compliments, casual conversation.
            OUTPUT:
            {{"type": "small_talk", "response": "Your friendly reply here"}}

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


        self.router_chain = (
            PromptTemplate.from_template(router_template) 
            | self.llm 
            | JsonOutputParser()
        )

        refine_template = """
            Given a chat history and a follow-up question, rephrase the question to be standalone and specific.
            If the history is empty or irrelevant, return the question as is.
            
            Chat History:
            {chat_history}
            
            Follow Up Input: {question}
            
            Standalone Question:

        """

        self.refine_chain = (
            PromptTemplate.from_template(refine_template) 
            | self.llm 
            | StrOutputParser()
        )

    

    """
    *   @brief Process the user query with context to generate an appropriate response.

    *   @param user_query: Contains the user's input query string.
    *   @param chat_history_str: Contains the chat history as a single string compressed by the Agent.
    *   @param user_obj: Contains user information, including authentication status.
    
    *   @pre  user_obj must be a valid dictionary with user details, including authentication status.
    *   @pre  user_query must be a valid user question string.
    
    *   @return A dictionary containing the generated response, its source, and the original query.
    *   @post  Guarantees that response is created based on the context based user query and chat history.

    *   @throws "I couldn't generate an answer.", if no valid information is found in the KB.
    *   @throws "I encountered an error processing your request.", if agent is not able to create a valid response.

    """


    def process_query(self, user_query, chat_history_str, user_obj):
        try:
            # ---------- ORDER ID DETECTION ----------
            order_match = ORDER_ID_RE.search(user_query)

            # User intent is order-related BUT no valid 9-digit ID
            if any(k in user_query.lower() for k in ["order", "track", "tracking"]) and not order_match:
                return {
                    "answer": (
                        "Please enter a valid **9-digit Order ID**.\n\n"
                        "Example: `696292323`"
                    ),
                    "source": "order_tracking",
                    "original_query": user_query
                }

            # ---------- ORDER FOUND ----------
            if order_match:
                order_token = order_match.group(3).strip()
                print("[DEBUG] order token:", order_token)

                order = find_order_in_db(order_token)
                print("[DEBUG] order found:", bool(order))

                if not order:
                    return {
                        "answer": (
                            f"I couldn’t find any order with ID **{order_token}**. "
                            "Please check the number."
                        ),
                        "source": "client_orders",
                        "original_query": user_query
                    }

                intent_result = classify_order_intent(self.llm, user_query)
                intent = intent_result.get("intent", "BASIC_ORDER_INFO")

                filtered_order = apply_data_policy(order, intent)

                prompt = (
                    DETAILED_RESPONSE_PROMPT
                    if intent == "DETAILED_ORDER_INFO"
                    else BASIC_RESPONSE_PROMPT
                )

                response = self.llm.invoke(
                    prompt.format(order_data=json.dumps(filtered_order, indent=2))
                )

                return {
                    "answer": response.content,
                    "source": "client_orders",
                    "original_query": user_query
                }

            # ---------- NORMAL ROUTING ----------
            route = self.router_chain.invoke({"query": user_query})

            if isinstance(route, dict) and route.get("type") == "small_talk":
                return {
                    "answer": route.get("response", "Hello! How can I help you with ShipCube?"),
                    "source": "small_talk",
                    "original_query": user_query
                }

            if isinstance(route, dict) and route.get("type") == "pricing":
                if user_obj.get("is_guest", False):
                    return {
                        "answer": (
                            "This query involves detailed pricing information. "
                            "Please **log in** to view exact charges and rates."
                        ),
                        "source": "auth_required",
                        "original_query": user_query
                    }

            refined_query = self.refine_chain.invoke({
                "chat_history": chat_history_str,
                "question": user_query
            })

            print(f"[Agent] Original: {user_query} | Refined: {refined_query}")

            rag_response = generate_answer_from_retrieval(refined_query) or {}

            sources = (
                ", ".join(s["source"] for s in rag_response.get("sources", []))
                if "sources" in rag_response
                else "knowledge_base"
            )

            return {
                "answer": rag_response.get("answer", "I don't know."),
                "source": sources,
                "original_query": refined_query
            }

        except Exception as e:
            print(f"[Agent Error] {e}")
            return {
                "answer": "I encountered an error processing your request.",
                "source": "error",
                "original_query": user_query
            }

shipcube_agent = RAGAgent()