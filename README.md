# RAG app: LangChain + Pinecone + OpenAI

```bash
pip install -r requirements.txt
cp .env.example .env        # add your keys
# put .pdf / .txt / .md files in ./data
python ingest.py            # add --reset to wipe the namespace first
python chatbot.py           # terminal chat
streamlit run app.py        # web chat
```
