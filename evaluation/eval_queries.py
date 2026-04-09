import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingest import load_model, get_qdrant_client, setup_collection
from app.retriever import retrieve_pages
from app.generator import generate_answer

#  Test queries covering all modalities 
TEST_QUERIES = [
    {"query": "What is the main topic of this document?", "type": "Text"},
    {"query": "Summarize the key findings", "type": "Text"},
    {"query": "What numbers or statistics are mentioned?", "type": "Table"},
    {"query": "What do the charts or figures show?", "type": "Chart"},
    {"query": "What are the main conclusions?", "type": "Text"},
    {"query": "What methodology is used?", "type": "Text"},
    {"query": "What are the recommendations?", "type": "Text"},
]

def run_evaluation():
    print("Loading model")
    model, processor = load_model()
    client = get_qdrant_client()

    print("EVALUATION RESULTS")

    results = []
    for i, test in enumerate(TEST_QUERIES):
        print(f"\nQuery {i+1} [{test['type']}]: {test['query']}")


        retrieved = retrieve_pages(test["query"], model, processor, client)

        if not retrieved:
            print(" No pages retrieved")
            results.append({"query": test["query"], "type": test["type"], "status": "FAILED", "pages": []})
            continue

        answer = generate_answer(test["query"], retrieved)
        pages = [f"Page {p['page_number']} (score: {p['score']:.3f})" for p in retrieved]

        print(f" Retrieved: {', '.join(pages)}")
        print(f"Answer preview: {answer[:200]}...")

        results.append({
            "query": test["query"],
            "type": test["type"],
            "status": "SUCCESS",
            "pages": pages,
            "answer_preview": answer[:200]
        })

    # Summary
    print("SUMMARY")
    success = sum(1 for r in results if r["status"] == "SUCCESS")
    print(f"Total queries: {len(results)}")
    print(f"Successful: {success}")
    print(f"Failed: {len(results) - success}")
    print(f"Accuracy: {success/len(results)*100:.1f}%")

if __name__ == "__main__":
    run_evaluation()