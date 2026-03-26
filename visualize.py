from dotenv import load_dotenv
load_dotenv()

from app.haystack.pipelines.chat_retrieval import create_chat_retrieval_pipeline

pipeline = create_chat_retrieval_pipeline()
pipeline.draw(path="chat_pipeline.png")
print("Done!")