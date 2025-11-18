import torch
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- setup ---
model_name = "gpt2"
tok = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()
torch.manual_seed(0)

model.config.use_cache = True

# --- prompt ---
ids = torch.tensor([[32, 13]])#, 198, 198]], dtype=torch.long)
attention_mask = torch.ones_like(ids)

# storage for captured activations
caps = {
    "ln1_block0": None,
    "attn_out_block0": None,
}

# --- hook for block 0 ln_1 ---
def ln1_hook(module, input, output):
    # output: [batch, seq, hidden]
    out0 = output.detach().cpu()
    caps["ln1_block0"] = out0

# --- hook for block 0 attention output ---
def attn_hook(module, input, output):
    # GPT-2 attention usually returns (attn_output, present, attn_weights?) or similar
    if isinstance(output, (tuple, list)):
        attn_out = output[0]
    else:
        attn_out = output
    caps["attn_out_block0"] = attn_out.detach().cpu()

hook_ln1 = model.transformer.h[0].ln_1.register_forward_hook(ln1_hook)
hook_attn = model.transformer.h[0].attn.register_forward_hook(attn_hook)

with torch.no_grad():
    # --- WTE ---
    wte_out = model.transformer.wte(ids)

    # --- WPE ---
    position_ids = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
    wpe_out = model.transformer.wpe(position_ids)

    # --- Input into block 0 ---
    embedded = wte_out + wpe_out

    # --- Forward pass (hooks trigger inside block 0) ---
    out = model(
        input_ids=ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )

    logits = out.logits  # final logits

# remove hooks
hook_ln1.remove()
hook_attn.remove()

# --- printing ---
print("=== WTE output ===")
print("Shape:", wte_out.shape)
print(wte_out)

print("\n=== WPE output ===")
print("Shape:", wpe_out.shape)
print(wpe_out)

print("\n=== WTE + WPE ===")
print("Shape:", embedded.shape)
print(embedded)

print("\n=== Block 0 ln_1 output (input to attention) ===")
print("Shape:", caps['ln1_block0'].shape)
print(caps["ln1_block0"])

print("\n=== Block 0 attention output (post-attention, pre-residual add / pre-ln_2) ===")
print("Shape:", caps['attn_out_block0'].shape)
print(caps["attn_out_block0"])

print("\n=== Final logits ===")
print("Shape:", logits.shape)
print(logits)

sys.exit(0)
