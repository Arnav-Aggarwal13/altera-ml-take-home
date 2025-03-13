import transformers as tr
import torch
import torch.nn.functional as F
import numpy as np


amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
alpha = 0.1
tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)

user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)

amateur_model = tr.AutoModelForCausalLM.from_pretrained(amateur_path)
expert_model = tr.AutoModelForCausalLM.from_pretrained(expert_path)


def get_next_token_probs(model, input_ids):
    """Computes the probability distribution over the next token."""
    input_tensor = torch.tensor([input_ids]).to(model.device)

    with torch.no_grad():
        outputs = model(input_tensor)

    # Get logits for the last token position
    logits = outputs.logits[:, -1, :]  # Shape: (1, vocab_size)

    # Convert logits to log probabilities
    probs = F.softmax(logits, dim=-1)

    return probs.cpu().numpy().flatten()
 
    
    


def contrastive_generation(amateur, expert, prompt, max_tokens, sampling_strategy='topk', k=5):
    # general approach
    # 1. generate amateur response
	# 2. generate expert response
    # 3. grab amateur logits, and exper lofits
    # 4. use plausibility constraint to filter our implausible tokens 
    # 4. use log_softmax to obtain log prob of each
	# 5. calculate contrastive loss
    # 6. based on this choose top k tokens
    # 7. sample from this k tokens, and repeat 
    
	updated_prompt = prompt
	for i in range(max_tokens):
		amateur_probs = get_next_token_probs(amateur, updated_prompt)
		expert_probs = get_next_token_probs(expert, updated_prompt)
        max_prob_expert = np.max(expert_probs)
		cd_obj = np.zeros_like(amateur_probs)
		for i, (exp_val, amt_val) in enumerate(zip(expert_probs, amateur_probs)):
			if exp_val >= (alpha * max_prob_expert):
				cd_obj[i] = np.log(exp_val) - np.log(amt_val)
			else:
				cd_obj[i] = float('-inf')
		

				
		


    
    
    
