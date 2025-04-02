# MY-OWN-LLM

Welcome to **MY-OWN-LLM**! This is my 7-day journey of building a language model from scratch, hitting around 90–100M parameters. I started with a basic custom LLM, made it efficient with ALBERT-like tricks, and fine-tuned TinyLlama with LoRA on Wikipedia data—all on Google Colab with a T4 GPU. It was tough—GPU limits, Colab timeouts, and lots of coffee—but I did it! Check out my story below, day by day.

---

## Project Overview

- **Goal**: Build my own LLM from scratch in 7 days, starting simple and ending with a fine-tuned TinyLlama model.
- **Tools**: Python, PyTorch, Transformers, LoRA, BitsAndBytes, Gradio, Google Colab (T4 GPU).
- **Dataset**: Tokenized Wikipedia data (`tokenized_wikitext`).
- **Final Output**: A fine-tuned TinyLlama model with a Gradio UI, generating text like "The history of science is often marked by a succession of crises…"

---
![Description of the image](https://raw.githubusercontent.com/manu14357/MY-OWN-LLM/refs/heads/main/Screenshot%202025-04-02%20131746.png)

## Day-by-Day Journey

### Day 1: Starting Simple
- **What I Did**: Wrote a basic custom LLM with `nn.TransformerEncoderLayer`, embeddings, and a linear output layer.
- **Parameters**: Around 90–100M (vocab_size=32000, d_model=768, num_layers=12).
- **Output**: Total gibberish—like "about about Wagner"—but it ran!
- **Challenges**: Figuring out tokenization and model setup.
- **LinkedIn Post**: [Day 1 Post](https://www.linkedin.com/posts/manu1435_day1of7llmjourney-machinelearning-ai-activity-7309991608031006721-_69S?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAADjcrkQBSEeDXyyLLO9JOr3MIWAXPdCzDJ8)
  
### Day 2: Training Begins
- **What I Did**: Added a `DataLoader` with my tokenized Wikipedia data and trained with `AdamW` and `CrossEntropyLoss`.
- **Code**: Used `batch[:, :-1]` for inputs and `batch[:, 1:]` for targets.
- **Output**: Loss started dropping, but text was still nonsense.
- **Challenges**: GPU memory crashes—had to tweak batch sizes.
- **LinkedIn Post**: [Day 2 Post](https://www.linkedin.com/posts/manu1435_day2of7llmjourney-machinelearning-ai-activity-7310334940376702977-81IE?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAADjcrkQBSEeDXyyLLO9JOr3MIWAXPdCzDJ8)
  
### Day 3: Adding Generation
- **What I Did**: Wrote a `generate_text` function with `torch.multinomial` and temperature sampling.
- **Output**: "Technology will shape agn impacts Febru sam…"—funny, but messy!
- **Challenges**: Getting coherent text was tough—needed more training.
- **LinkedIn Post**: [Day 3 Post](https://www.linkedin.com/posts/manu1435_day3of7llmjourney-machinelearning-ai-activity-7310702821790670848-0mk2?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAADjcrkQBSEeDXyyLLO9JOr3MIWAXPdCzDJ8)

### Day 4: Scaling Up
- **What I Did**: Increased layers and trained longer, saved checkpoints to `/content/drive/MyDrive/LLM/custom_llm_epoch_7.pth`.
- **Output**: Slightly better text, but still random.
- **Challenges**: Training took hours—Colab kept timing out!
- **LinkedIn Post**: [Day 4 Post](https://www.linkedin.com/posts/manu1435_day4of7llmjourney-machinelearning-ai-activity-7311068861276069892-ikSh?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAADjcrkQBSEeDXyyLLO9JOr3MIWAXPdCzDJ8)
  
### Day 5: Efficiency Boost
- **What I Did**: Rewrote my model as `EfficientCustomLLM` with shared layers (`nn.ModuleList([encoder_layer] * num_layers)`) and gradient checkpointing (`checkpoint_sequential`).
- **Parameters**: Dropped to 35.95M—way lighter!
- **Output**: Loss from 10.4889 to 10.1393, speed 70.33 examples/s.
- **Challenges**: Loading old weights into a new setup.
- **LinkedIn Post**: [Day 5 Post](https://www.linkedin.com/posts/manu1435_day5of7llmjourney-machinelearning-ai-activity-7312082341974196225-OxTe?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAADjcrkQBSEeDXyyLLO9JOr3MIWAXPdCzDJ8) 

### Day 6: Fine-Tuning Prep
- **What I Did**: Switched to TinyLlama (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`), added LoRA (`r=16`, `target_modules=["q_proj", "v_proj"]`), and quantized with `BitsAndBytesConfig`.
- **Output**: Model loaded, ready for fine-tuning.
- **Challenges**: Colab GPU limits hit again—had to optimize everything.
- **LinkedIn Post**: [Day 6 Post](https://www.linkedin.com/posts/manu1435_day6of7llmjourney-machinelearning-ai-activity-7312554194391994369-4zqp?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAADjcrkQBSEeDXyyLLO9JOr3MIWAXPdCzDJ8) 

### Day 7: Fine-Tuning & UI
- **What I Did**: Fine-tuned TinyLlama with `trainer.train()` (18+ Hours ), added a Gradio UI with `gr.Blocks`.
- **Output**: "How do these technologies help in predicting future events? 1. Predictive maintenance…"—not perfect, but so much better!
- **Challenges**: Took forever—GPU wasn't enough, hit Colab limits, switched accounts to finish.
- **LinkedIn Post**: [Day 7 Post](https://www.linkedin.com/placeholder-day7) 

---

## How to Run It

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/yourusername/MY-OWN-LLM.git
   cd MY-OWN-LLM
   ```

2. **Install Dependencies**:
   ```bash
   pip install transformers datasets torch sentencepiece accelerate bitsandbytes peft gradio
   ```

3. **Set Up Google Drive**:
   * Upload tokenized_wikitext to /content/drive/MyDrive/LLM/ in Colab.

4. **Run the Code**:
   * Open LLM.ipynb in Colab.
   * Update HF_TOKEN with your Hugging Face token.
   * Run all cells to fine-tune and launch the Gradio UI.

5. **Try It**:
   * Use the Gradio link (e.g., https://36fe5ab037a9c59c9d.gradio.live) to test prompts!

## Results

* **Start**: Gibberish like "about about Wagner."
* **End**: "How do these technologies help in predicting future events? 1. Predictive maintenance…"—okay-ish, but mine!
* **Stats**: Fine-tuning took 2+ Hours, loss from 12.9230 to 12.9763, 0.2044% trainable params with LoRA.

## Lessons Learned

* Building an LLM is hard but fun!
* Colab's free T4 GPU is great, but limits hit fast—timeouts and memory issues are real.
* LoRA and quantization save the day for efficiency.
* Patience is key— 6+ Days felt like forever!

## Next Steps

* Train longer with more data for better accuracy.
* Add more features to the Gradio UI.
* Maybe try a bigger model (if I get Colab Pro!).

## Full Code

Check out the complete notebook: [LLM.ipynb](./LLM.ipynb)

Happy coding, and thanks for following my journey! 🚀
