import pinecone
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("Ghani-25/LF_enrich_sim", device='cpu')
pinecone.init(api_key='026cfe1e-78ef-4fbd-8398-e6dcadc030e0', environment='asia-southeast1-gcp')
index = pinecone.Index('aipros')

def enrichir(query, count):
    xq = model.encode(query).tolist()
    result = index.query(xq, top_k=count, includeMetadata=False)
    res = result.to_dict() #conversion to dict
    lis = list(res.values())[0]
    return lis