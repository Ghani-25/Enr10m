import pinecone
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("Ghani-25/LF_enrich_sim", device='cpu')
pinecone.init(api_key='564016db-e934-4d4d-b2ed-1d5fbc1bf8b7', environment='northamerica-northeast1-gcp')
index = pinecone.Index('ai-prospects-finder')

def enrichir(query, count):
    xq = model.encode(query).tolist()
    result = index.query(xq, top_k=count, includeMetadata=False)
    res = result.to_dict() #conversion to dict
    lis = list(res.values())[0]
    return lis
