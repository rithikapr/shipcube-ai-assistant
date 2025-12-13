import pytest
from unittest.mock import patch
from utils.rag_pipeline import RAGAgent


"""
    @brief Simple mock replacement for the `.invoke()` interface used by the chains.

"""
class _MockChain:
    def __init__(self, behaviour):
        self._behaviour = behaviour

    def invoke(self, payload):
        return self._behaviour(payload)



"""
    @result Return a sample guest user object (not authenticated).

"""
@pytest.fixture
def sample_user_guest():
    return {"is_guest": True}


"""
    @result Return a sample logged-in user object (authenticated).

"""
@pytest.fixture
def sample_user_logged_in():
    return {"is_guest": False}


"""
    @result Return a sample small chat history.

"""
@pytest.fixture
def sample_chat_history():
    return "User: Hi\nAssistant: Hello, how can I help?\n"


"""
    @unit Checks if the small talk routing works as expected.

    @result When the router classifies input as small_talk, RAGAgent should return its intended response directly.

"""
def test_small_talk_routing_returns_chat_response(sample_chat_history):
    agent = RAGAgent()
    agent.router_chain = _MockChain(lambda payload: {"type": "small_talk", "response": "Hey there!"})
    agent.refine_chain = _MockChain(lambda payload: "stub")

    out = agent.process_query("hi", sample_chat_history, {"is_guest": False})

    assert isinstance(out, dict)
    assert out["source"] == "small_talk"
    assert out["answer"] == "Hey there!"
    assert out["original_query"] == "hi"


"""
    @unit Checks if the pricing guardrail routing works as expected for guest users.

    @result When the router classifies input as 'pricing' and the user is a guest, RAGAgent should prompt authentication and not call retrieval.

"""
def test_pricing_guardrail_for_guest_users(sample_chat_history, sample_user_guest):
    agent = RAGAgent()
    agent.router_chain = _MockChain(lambda payload: {"type": "pricing"})
    agent.refine_chain = _MockChain(lambda payload: "should-not-be-used")

    out = agent.process_query("How much does warehousing cost?", sample_chat_history, sample_user_guest)

    assert out["source"] == "auth_required"
    assert "log in" in out["answer"].lower()
    assert out["original_query"] == "How much does warehousing cost?"


"""
    @unit Checks if the pricing guardrail routing works as expected for logged in users.

    @result When the router classifies input as 'pricing' and the user is logged in, RAGAgent should call retrieval with the user's context.

"""
def test_refinement_and_retrieval_are_invoked(sample_chat_history, sample_user_logged_in):
    agent = RAGAgent()
    agent.router_chain = _MockChain(lambda payload: {"type": "technical"})

    def _refine(payload):
        assert "chat_history" in payload and "question" in payload
        return "refined: what is warehousing cost per pallet?"

    agent.refine_chain = _MockChain(_refine)
    fake_rag_response = {
        "answer": "Approx. $5 per pallet per day.",
        "sources": [{"source": "kb/warehousing-pricing"}]
    }

    with patch("utils.rag_pipeline.generate_answer_from_retrieval", return_value=fake_rag_response) as mocked_retrieval:
        out = agent.process_query("pricing for warehousing?", sample_chat_history, sample_user_logged_in)
        mocked_retrieval.assert_called_once_with("refined: what is warehousing cost per pallet?")

    assert out["answer"] == fake_rag_response["answer"]
    assert "kb/warehousing-pricing" in out["source"]
    assert out["original_query"] == "refined: what is warehousing cost per pallet?"


"""
    @unit Checks if the exception handling works as expected.

    @result When the router receives an error, RAGAgent should prompt the pre-defined error response.

"""
def test_agent_handles_exceptions_gracefully(sample_chat_history):
    agent = RAGAgent()
    def _raise(payload):
        raise RuntimeError("simulated failure")

    agent.router_chain = _MockChain(_raise)
    agent.refine_chain = _MockChain(lambda payload: "unused")

    out = agent.process_query("this will error", sample_chat_history, {"is_guest": False})

    assert out["source"] == "error"
    assert "encountered an error" in out["answer"].lower()
    assert out["original_query"] == "this will error"


"""
    @integration Checks end to end behaviour with realistic stubs for chains and retrieval.

    @result When the router receives a user_query, RAGAgent should prompt the defined context driver routed response.

"""
def test_integration_like_flow_with_realistic_stubs(sample_chat_history, sample_user_logged_in):
    agent = RAGAgent()
    def router_behaviour(payload):
        query = payload.get("query", "")
        if "hello" in query.lower():
            return {"type": "small_talk", "response": "Hello friend!"}
        if "price" in query.lower():
            return {"type": "pricing"}
        return {"type": "technical"}

    agent.router_chain = _MockChain(router_behaviour)
    agent.refine_chain = _MockChain(lambda payload: payload.get("question", "").strip().rstrip("?") + "?")

    response_payload = {
        "answer": "Here is the technical explanation.",
        "sources": [{"source": "kb/technical-docs"}, {"source": "kb/faq"}]
    }

    with patch("utils.rag_pipeline.generate_answer_from_retrieval", return_value=response_payload):
        out = agent.process_query("Explain how n=1 routing works", sample_chat_history, sample_user_logged_in)

    assert out["answer"] == response_payload["answer"]
    assert "kb/technical-docs" in out["source"]
    assert out["original_query"].endswith("?")
