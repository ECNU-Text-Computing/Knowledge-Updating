from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import re

def extract_fact(text):
    if text and not text.endswith('.'):
        text += '.'
    # Regex pattern to match the fact after the specified phrases, accounting for both quoted and non-quoted facts
    pattern = r'The fact that best matches the (question|core knowledge asked in the question|core knowledge in the question) is\s*[:\s]*["]?(.*?)(?=[".\n])'

    match = re.search(pattern, text)

    if match:
        fact = match.group(2).strip()

        if fact and not fact.endswith('.'):
            fact += '.'

        return fact
    else:
        return text

class SummaryTool():
    def __init__(self, summary_model, summary_tok, device):
        self.summary_model = summary_model
        self.summary_tok = summary_tok
        self.device = device

    def summary(self, question, retrieved_facts, summary_prompt, max_new_tokens=50):
        retrieved_facts = "\n".join(retrieved_facts)
        # 倒序排列
        # retrieved_facts = '\n'.join(retrieved_facts[::-1])

        prompt = summary_prompt.format(retrieved_facts=retrieved_facts, question=question)

        ids = self.summary_tok.encode(prompt, add_special_tokens=True)
        input_ids = torch.LongTensor([ids]).to(self.device)

        generated_ids = self.summary_model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.summary_tok.eos_token_id,
            do_sample=False,

        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)
        ]
        relevance = self.summary_tok.decode(generated_ids[0].detach().cpu().numpy(), skip_special_tokens=True)

        if "no relevant information" in relevance.lower() or "no relevant fact" in relevance.lower() or "false" in relevance.lower():

            return None
        else:
            relevance = extract_fact(relevance)
            # if "Explanation" in relevance:
            #    relevance = relevance.split("Explanation")[0].strip()
            return relevance