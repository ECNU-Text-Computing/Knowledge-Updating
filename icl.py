
class EditTool():
    def __init__(self, model, tok, device):

        self.model = model
        self.tok = tok
        self.device = device

    def single_edit(self, prompt, target_new, knowledge=None, locality=False, vanilla_genration=True,
                    answer_prompt=None, eval_metric="token_em"):

        if eval_metric == "contain":
            if isinstance(target_new, str):
                target_new = [target_new]

            if knowledge is None:

                question = prompt

                # target_new_tokens = self.tok.encode(target_new, add_special_tokens=False)
                prompt_tok = self.tok(
                    question,
                    return_tensors="pt",
                ).to(self.device)
                gen_token = self.model.generate(
                    input_ids=prompt_tok['input_ids'],
                    attention_mask=prompt_tok['attention_mask'],
                    max_new_tokens=30,
                    pad_token_id=self.tok.eos_token_id,
                    do_sample=False,

                )
                out_answer_tokens = gen_token.detach().cpu().numpy().tolist()[0]

                input_length = prompt_tok['input_ids'].shape[1]  # 获取输入 token 的长度
                generated_tokens = gen_token.detach().cpu().numpy().tolist()[0][input_length:]  # 切片，获取生成的 token
                answer = self.tok.decode(generated_tokens, skip_special_tokens=True)

                result = max([float(target_new_answer.lower() in answer.lower()) for target_new_answer in target_new])

            else:
                question = prompt

                prompt = answer_prompt.format(question=question, knowledge=knowledge)

                prompt_tok = self.tok(
                    prompt,
                    return_tensors="pt",
                ).to(self.device)
                gen_token = self.model.generate(
                    input_ids=prompt_tok['input_ids'],
                    attention_mask=prompt_tok['attention_mask'],
                    max_new_tokens=30,
                    pad_token_id=self.tok.eos_token_id,
                    do_sample=False,

                )

                input_length = prompt_tok['input_ids'].shape[1]  # 获取输入 token 的长度
                generated_tokens = gen_token.detach().cpu().numpy().tolist()[0][input_length:]  # 切片，获取生成的 token
                answer = self.tok.decode(generated_tokens, skip_special_tokens=True)

                result = max([float(target_new_answer.lower() in answer.lower()) for target_new_answer in target_new])

            return result