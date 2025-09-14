import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- setup ---
model_name = "gpt2"
tok = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()
torch.manual_seed(0)

# enable KV cache for fast autoregressive decoding
model.config.use_cache = True

# --- prompt ---
prompt = "A"
inputs = tok(prompt, return_tensors="pt", add_special_tokens=False)
input_ids = inputs["input_ids"]          # [B=1, S0]
attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

# --- capture dict + hook ---
caps = {"ln_f_out_steps": []}

def ln_f_hook(_, inp, out):
    out0 = out[0] if isinstance(out, (tuple, list)) else out
    # clone to CPU so it won't be mutated when we keep decoding
    caps["ln_f_out_steps"].append(out0.detach().cpu())

h = model.transformer.ln_f.register_forward_hook(ln_f_hook)

# --- decoding config ---
max_new_tokens = 32
temperature = 0.0        # 0 = greedy. Set >0 to sample
top_k = None             # set e.g. 50 for top-k sampling
eos_id = tok.eos_token_id

generated = input_ids.clone()
past_key_values = None

with torch.no_grad():
    # first forward pass can take the whole prompt
    out = model(
        input_ids=generated,
        attention_mask=attention_mask,
        use_cache=True,
        past_key_values=past_key_values,
        return_dict=True,
    )
    past_key_values = out.past_key_values

    for step in range(max_new_tokens):
        # logits for the last position
        next_token_logits = out.logits[:, -1, :]  # [1, vocab]
        if temperature and temperature > 0:
            logits = next_token_logits / temperature
            if top_k is not None and top_k > 0:
                # top-k filtering
                topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(dim=-1, index=topk_idx, src=topk_vals)
                probs = torch.nn.functional.softmax(filtered, dim=-1)
            else:
                probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1,1]
        else:
            # greedy
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [1,1]

        # append
        generated = torch.cat([generated, next_token], dim=-1)

        # early stop on EOS if present
        if eos_id is not None and next_token.item() == eos_id:
            break

        # incremental forward: feed only the new token, with cache
        out = model(
            input_ids=next_token,                 # only the last token
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
        )
        past_key_values = out.past_key_values

# remove hook
h.remove()

# --- outputs ---
# decode full string
decoded = tok.decode(generated[0], clean_up_tokenization_spaces=False)

print("=== Generated text ===")
print(decoded)
