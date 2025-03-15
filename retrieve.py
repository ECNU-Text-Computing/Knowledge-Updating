import torch

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from prepare_requests import load_data, _prepare_requests


class RetrieveTool():
    def __init__(self, retriever_type, retriever_path, retriever, retriever_tok, device):
        if retriever_type == "ance":
            self.ance_model = SentenceTransformer(retriever_path)
        else:
            self.retriever = retriever
            self.retriever_tok = retriever_tok
        self.retriever_type = retriever_type
        self.device = device

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def get_sent_embeddings(self, memory, BSZ=32):
        if self.retriever_type == "ance":
            # Use ANCE model to encode sentences in memory
            self.memory_embedding = torch.tensor(self.ance_model.encode(memory)).to(self.device)
        else:
            all_embs = []
            for i in tqdm(range(0, len(memory), BSZ)):
                sent_batch = memory[i:i + BSZ]
                inputs = self.retriever_tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to(
                    self.device)
                with torch.no_grad():
                    outputs = self.retriever(**inputs)
                    embeddings = self.mean_pooling(outputs[0], inputs['attention_mask'])
                all_embs.append(embeddings.cpu())
            all_embs = torch.vstack(all_embs)
            self.memory_embedding = all_embs
        # self.memory_embedding = all_embs / all_embs.norm(dim=1, keepdim=True)  # L2 归一化

    def retrieve(self, query, memory, k):
        if self.retriever_type == "ance":
            # Use ANCE model to encode the query
            query_emb = torch.tensor(self.ance_model.encode([query])).to(self.device)
        else:
            inputs = self.retriever_tok([query], padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.retriever(**inputs)
                query_emb = self.mean_pooling(outputs[0], inputs['attention_mask']).cpu()

        # query_emb = query_emb / query_emb.norm()  # 对查询嵌入进行 L2 归一化

        sim = (query_emb @ self.memory_embedding.T)[0]
        knn = sim.topk(k, largest=True)
        fact_ids = knn.indices

        # Retrieve multiple facts
        retrieved_facts = [memory[fact_id] for fact_id in fact_ids]
        similarity_scores = knn.values.detach().cpu().numpy().tolist()

        return retrieved_facts, similarity_scores  # return a list