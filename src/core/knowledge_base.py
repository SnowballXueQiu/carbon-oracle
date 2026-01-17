import os
import sys
import contextlib
import chromadb

KB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "knowledge_base")

@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

class KnowledgeBase:
    def __init__(self):
        # Persistent storage
        with suppress_stderr():
            self.client = chromadb.PersistentClient(path=KB_PATH)
            self.collection = self.client.get_or_create_collection(name="experiments_rag")

    def add_experiment_insight(self, batch_id: str, summary_text: str, result_quality: str):
        """
        Stores the qualitative analysis of an experiment for future retrieval.
        quality: 'good' | 'bad' | 'abnormal'
        """
        with suppress_stderr():
            self.collection.upsert(
                documents=[summary_text],
                metadatas=[{"batch_id": batch_id, "quality": result_quality}],
                ids=[batch_id]
            )
        print(f"[RAG] Indexed analysis for {batch_id} into Vector DB.")

    def find_similar_cases(self, current_conditions_text: str, n_results: int = 2) -> str:
        """
        Retrieves context from past similar experiments to help the AI Advisor.
        """
        try:
            with suppress_stderr():
                results = self.collection.query(
                    query_texts=[current_conditions_text],
                    n_results=n_results
                )
            
            if not results["documents"] or not results["documents"][0]:
                return "No similar historical cases found."
                
            context = "Historical Precedents:\n"
            metadatas = results.get("metadatas")
            if not metadatas or not metadatas[0]:
                return "No metadata found for similar historical cases."
            for i, doc in enumerate(results["documents"][0]):
                meta = metadatas[0][i]
                context += f"- Batch {meta['batch_id']} ({meta['quality']}): {doc}\n"
            
            return context
        except Exception as e:
            return f"Vector DB Error: {e}"
