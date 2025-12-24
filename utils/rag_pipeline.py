import json
import sqlite3
from utils.ai_model import (
    classify_order_intent,
    BASIC_RESPONSE_PROMPT,
    DETAILED_RESPONSE_PROMPT
)
from utils.rag_model.llm import llm
from utils.rag_model.prompts import (
    router_chain,
    refine_chain,
    rag_answer_chain,
    small_talk_chain
)
from utils.config import (
    ORDER_ID_RE,
    BASIC_FIELDS,
    DETAILED_FIELDS,
    ZIP_RE,
    WEIGHT_RE
)
from utils.rag_model.retrieval import (
    vectorstore_retrieval,
    get_user_intent,
    get_refined_query,
    get_rag_answer,
    small_talk_response
)
from utils.rate_engine import find_best_rate

def wants_rate_explanation(llm, user_query: str) -> bool:
    """
    Returns True only if the user is explicitly asking
    for an explanation of how the shipping rate was calculated.
    """
    prompt = f"""
You are an intent classifier.

User query:
"{user_query}"

Decide whether the user is explicitly asking for
HOW the shipping cost was calculated or WHY a carrier was chosen.

Rules:
- Reply ONLY with TRUE or FALSE
- TRUE only if the user wants calculation steps, reasoning, or explanation
- FALSE otherwise
"""
    try:
        resp = llm.invoke(prompt).content.strip().upper()
        return resp == "TRUE"
    except Exception:
        return False

def build_hub_price_summary(rate_result: dict) -> str:
    """
    Builds a single customer-facing sentence showing
    cheapest carrier + price for each hub.
    """
    parts = []

    for hub_data in rate_result.get("per_hub_results", []):
        hub = hub_data["hub"]
        carrier = hub_data["best_carrier"]
        price = hub_data["best_price"]

        parts.append(f"{carrier} from {hub} hub at ${price}")

    joined = ", ".join(parts)

    return (
        f"For your {rate_result['effective_weight_lb']} lb package to ZIP "
        f"{rate_result['input_zipcode']}, the cheapest options are {joined}."
    )


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
        self.llm = llm
        self.router_chain = router_chain
        self.refine_chain = refine_chain
        self.rag_answer_chain = rag_answer_chain
        self.small_talk_chain = small_talk_chain


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
    
    def process_query(self, user_query, chat_history_str, user_obj, last_rate_result=None):
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
                       
            # ---------- RATE (ZIP + WEIGHT) DETECTION ----------
            zip_match = ZIP_RE.search(user_query)
            weight_match = WEIGHT_RE.search(user_query)

            # ZIP only
            if zip_match and not weight_match:
                return {
                    "answer": "Please provide the **package weight** to calculate shipping rates.",
                    "source": "input_required",
                    "original_query": user_query
                }

            # Weight only
            if weight_match and not zip_match:
                return {
                    "answer": "Please provide the **destination ZIP code** to calculate shipping rates.",
                    "source": "input_required",
                    "original_query": user_query
                }

            if zip_match and weight_match:

                if user_obj.get("is_guest", False):
                    return {
                        "answer": (
                            "Shipping rate estimates are available only for logged-in users.\n\n"
                            "Please **log in** to view pricing."
                        ),
                        "source": "auth_required",
                        "original_query": user_query
                    }

                zipcode = int(zip_match.group())
                weight_val = float(weight_match.group(1))
                unit = weight_match.group(2).lower()

                if unit == "kg":
                    weight_val *= 2.20462
                elif unit == "oz":
                    weight_val /= 16

                #  1. Always calculate rate
                rate_result = find_best_rate(zipcode, weight_val)

                if "error" in rate_result:
                    return {
                        "answer": rate_result["error"],
                        "source": "rate_engine",
                        "original_query": user_query
                    }

                #  2. CHECK IF EXPLANATION IS EXPLICITLY ASKED
                explain_requested = wants_rate_explanation(self.llm, user_query)

                if explain_requested:
                    explanation = self.llm.invoke(f"""
            You are ShipCube AI.

            Explain how the shipping rate was calculated in clear, customer-friendly sentences.

            Strict rules:
            - Use the per_hub_results field from the data
            - Explain pricing for EACH hub present
            - Do NOT skip any hub
            - After explaining each hub, explain how the final rate was selected
            - No formulas
            - No JSON
            - No bullet points
            - Do NOT invent data

            DATA:
            {json.dumps(rate_result, indent=2)}
            """)

                    return {
                        "answer": explanation.content,
                        "source": "rate_engine",
                        "original_query": user_query
                    }

                #  3. DEFAULT  ONE-LINE SUMMARY ONLY
                summary_sentence = build_hub_price_summary(rate_result)

                user_obj["last_rate_result"] = rate_result

                return {
                    "answer": summary_sentence,
                    "source": "rate_engine",
                    "original_query": user_query,
                    "rate_data": rate_result
                }
            
            # ---------- FOLLOW-UP RATE EXPLANATION (NO ZIP/WEIGHT IN QUERY) ----------
            if wants_rate_explanation(self.llm, user_query):
                last_rate = user_obj.get("last_rate_result")

                if last_rate:
                    explanation = self.llm.invoke(f"""
            You are ShipCube AI.

            Explain how the shipping rate was calculated in clear, customer-friendly sentences.

            Strict rules:
            - Use the per_hub_results field from the data
            - Explain pricing for EACH hub present
            - Do NOT skip any hub
            - After explaining each hub, explain how the final rate was selected
            - No formulas
            - No JSON
            - No bullet points
            - Do NOT invent data

            DATA:
            {json.dumps(last_rate, indent=2)}
            """)

                    return {
                        "answer": explanation.content,
                        "source": "rate_engine",
                        "original_query": user_query
                    }

            # ---------- NORMAL ROUTING ----------
            route = get_user_intent(user_query)
            refined_query = get_refined_query(chat_history_str, user_query)

            # print(f"[Agent] Route: {route}")
            # print(f"[Agent] Original: {user_query} | Refined: {refined_query}")

            if route == "small_talk":
                response = small_talk_response(refined_query)

                return {
                    "answer": response or "Hello! How can I help you with ShipCube?",
                    "source": "small_talk",
                    "original_query": user_query
                }

            if route == "pricing":
                if user_obj.get("is_guest", False):
                    return {
                        "answer": (
                            "This query involves detailed pricing information. "
                            "Please **log in** to view exact charges and rates."
                        ),
                        "source": "auth_required",
                        "original_query": user_query
                    }

            # print(f"[Agent] Refined Query: {refined_query}")

            retrieved_chunks, metadata = vectorstore_retrieval(refined_query, top_k=3, threshold=0.33)
            # print(f"[Agent] Retrieved Chunks: {retrieved_chunks}")
            # print(f"[Agent] Metadata: {metadata}")

            rag_response = get_rag_answer(refined_query, retrieved_chunks, metadata)

            sources = (
                "faq_semantic"
                if retrieved_chunks else "none"
            )

            return {
                "answer": rag_response.get("answer") or "I couldn't obtain valid information. Could you please detail your question?",
                "source": sources,
                "original_query": user_query
            }

        except Exception as e:
            print(f"[Agent Error] {e}")
            return {
                "answer": "I encountered an error processing your request.",
                "source": "error",
                "original_query": user_query
            }

shipcube_agent = RAGAgent()