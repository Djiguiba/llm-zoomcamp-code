import sys
import os

current_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, root_dir)

from dotenv import load_dotenv
from anthropic import Anthropic

from db_save import save_conversation
from utils.ingest import load_faq_data, build_index
from utils.metrics import RAGWithMetrics

def create_assistant():
    load_dotenv()


    documents = load_faq_data()
    index = build_index(documents)

    return RAGWithMetrics(
        index=index,
        llm_client=Anthropic()
    )


if __name__ == "__main__":
    assistant = create_assistant()

    query = "How do I get a certificate?"
    if len(sys.argv) > 1:
        query = sys.argv[1]
    
    answer = assistant.rag(query)
    save_conversation(assistant.last_call, query, "llm-zoomcamp")
    print(answer)
