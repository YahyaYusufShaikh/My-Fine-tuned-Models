# 🚀 Docker Assistant (LLM Fine-Tuned Model)

An intelligent AI assistant fine-tuned to understand and generate **Docker commands and explanations** using natural language.

This project uses a fine-tuned LLM (LoRA + DPO) trained on Docker-related datasets and deployed with an interactive UI using Gradio.

---

## 🧠 Features

* 🔹 Convert natural language → Docker commands
* 🔹 Explain Docker concepts in simple terms
* 🔹 Supports conversational prompts
* 🔹 Fine-tuned using LoRA + DPO
* 🔹 Interactive UI with Gradio
* 🔹 Deployable on Hugging Face Spaces

---

## 🏗️ Model Details

* **Base Model:** LLaMA 3.2 (1B)
* **Fine-Tuning:** LoRA (Low-Rank Adaptation)
* **Alignment:** DPO (Direct Preference Optimization)
* **Datasets Used:**

  * dockerNLcommands
  * Human-Like-DPO-Dataset

---

## ⚙️ Tech Stack

* Python
* Transformers
* PyTorch
* PEFT (LoRA)
* TRL (SFT + DPO)
* LoRA
* Gradio
* BitsAndBytes (8-bit quantization)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YahyaYusuf/First-AI-Agent.git
cd docker-assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
python app.py
```

Then open:

```
http://127.0.0.1:7860
```

---

## 🌐 Deployment (Hugging Face Spaces)

1. Create a new Space on Hugging Face
2. Select **Gradio** as SDK
3. Upload:

   * `app.py`
   * `requirements.txt`
4. Set hardware to **T4 GPU (recommended)**
5. Your app will auto-deploy 🚀

---

## 🧪 Example Prompts

* "How do I list all running Docker containers?"
* "Explain what a Dockerfile is."
* "How to stop all running containers?"
* "Difference between docker run and docker exec"

---

## 🎛️ Generation Parameters

| Parameter   | Description              |
| ----------- | ------------------------ |
| Temperature | Controls randomness      |
| Top-p       | Controls diversity       |
| Max Tokens  | Controls response length |

---

## 📁 Project Structure

```
.
├── app.py                # Gradio UI + inference
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
└── results/             # Training outputs (optional)
```

---

## ⚠️ Limitations

* May generate incorrect commands in edge cases
* Requires GPU for optimal performance
* Large model → slower inference on CPU

---

## 🚀 Future Improvements

* Add Docker command validation
* Integrate CLI tool version
* Add memory/chat history
* Fine-tune on larger datasets

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a new branch
3. Submit a PR

---

## 👨‍💻 Author

**Yahya Yusuf**

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
