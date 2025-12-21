from utils.rag_model.retrieval import vectorstore_retrieval

retrieved_chunks, metadata = vectorstore_retrieval("Query: What other services does ShipCube offer?. Context: ShipCube simplifies shipping for businesses, managing orders, rates, and tracking. User asked about services, warehouses, and other services.", top_k=3, threshold=0.5)
print(f"[Agent] Retrieved Chunks: {retrieved_chunks}")
print(f"[Agent] Metadata: {metadata}")