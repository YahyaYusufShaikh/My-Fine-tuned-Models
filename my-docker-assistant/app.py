"""
STEP 2 — Gradio App: Users type a prompt, model generates a response.
Deploy this on Hugging Face Spaces (free!) — see README for instructions.
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# If you pushed your merged model to HF Hub, use your repo ID here.
# If you want to test locally before pushing, use MERGED_SAVE_DIR path.
MODEL_ID = "yahya2004/Llama3.2-Docker"

# Use 8-bit quantization to save GPU memory (works on free Spaces T4 GPU)
USE_QUANTIZATION = True
# ──────────────────────────────────────────────────────────────────────────────


def load_model():
    print("Loading model...")
    quant_config = BitsAndBytesConfig(load_in_8bit=True) if USE_QUANTIZATION else None

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded!")
    return model, tokenizer


# Load once at startup
model, tokenizer = load_model()


def generate_response(
    user_prompt: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """
    Formats the prompt using the model's chat template and generates a response.
    Works for both LLaMA and Qwen models — they both support the messages format.
    """
    if not user_prompt.strip():
        return "Please enter a prompt."

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt.strip()})

    # Apply the model's built-in chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the input)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


# ─── GRADIO UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="My Fine-Tuned Model", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# My Fine-Tuned Model")
    gr.Markdown(
        "This model was fine-tuned using **LoRA + DPO** on the "
        "`dockerNLcommands` and `Human-Like-DPO-Dataset` datasets."
    )

    with gr.Row():
        with gr.Column(scale=3):
            system_box = gr.Textbox(
                label="System Prompt (optional)",
                placeholder="e.g. You are a helpful Docker command assistant.",
                lines=2,
            )
            user_box = gr.Textbox(
                label="Your Prompt",
                placeholder="Type something here...",
                lines=5,
            )
            submit_btn = gr.Button("Generate", variant="primary")
            output_box = gr.Textbox(label="Model Response", lines=10, interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("### Generation Settings")
            max_tokens = gr.Slider(64, 512, value=256, step=32, label="Max New Tokens")
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")

    submit_btn.click(
        fn=generate_response,
        inputs=[user_box, system_box, max_tokens, temperature, top_p],
        outputs=output_box,
    )

    # Example prompts
    gr.Examples(
        examples=[
            ["How do I list all running Docker containers?", "You are a helpful Docker command assistant."],
            ["What is the difference between Docker run and Docker exec?", ""],
            ["Explain what a Dockerfile is.", ""],
        ],
        inputs=[user_box, system_box],
    )

if __name__ == "__main__":
    demo.launch(share=True)   # share=True gives a public URL on Colab