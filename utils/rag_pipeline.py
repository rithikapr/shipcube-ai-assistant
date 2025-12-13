from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from utils.ai_model import generate_answer_from_retrieval
import dotenv


dotenv.load_dotenv()



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
            Classify the User Input into one of two categories:
            
            1. "small_talk": For greetings, compliments, "how are you", or off-topic chitchat.
            OUTPUT: {{"type": "small_talk", "response": "Your friendly, conversational response here."}}

            2. "pricing": Questions about costs, fees, rates, charges, invoices, or billing.
            OUTPUT: {{"type": "pricing"}}
            
            3. "technical": For questions about ShipCube services, pricing, warehousing, logistics concepts, or specific facilities.
            OUTPUT: {{"type": "technical"}}
            
            User Input: {query}
            Output JSON only:
            {
                "type": "routing"
            }
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
            route = self.router_chain.invoke({"query": user_query})
            
            # Robust check for small_talk type
            if isinstance(route, dict) and route.get('type') == 'small_talk':
                return {
                    "answer": route.get('response', "Hello! How can I help you with ShipCube?"),
                    "source": "small_talk",
                    "original_query": user_query
                }

            # ROUTE 2: PRICING GUARDRAIL
            if isinstance(route, dict) and route.get('type') == 'pricing':
                if user_obj.get('is_guest', False):
                    return {
                        "answer": "This query involves detailed pricing information. Please **log in** to view exact charges, rates, and financial details.",
                        "source": "auth_required",
                        "original_query": user_query
                    }
                
            refined_query = self.refine_chain.invoke({
                "chat_history": chat_history_str,
                "question": user_query
            })
            
            print(f" [Agent] Original: {user_query} | Refined: {refined_query}")

            rag_response = generate_answer_from_retrieval(refined_query)
            
            if rag_response and "sources" in rag_response:
                sources = ", ".join([s['source'] for s in rag_response.get("sources", [])])
            else:
                sources = "knowledge_base"

            return {
                "answer": rag_response.get('answer', "I couldn't generate an answer."),
                "source": sources,
                "original_query": refined_query 
            }

        except Exception as e:
            print(f" [Agent Error] {e}")
            return {
                "answer": "I encountered an error processing your request.",
                "source": "error",
                "original_query": user_query
            }


shipcube_agent = RAGAgent()