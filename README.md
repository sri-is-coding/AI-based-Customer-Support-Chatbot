# Designing Intelligent Agents - Srivarshini Selvaraj 20512874
# AI-based Customer Support Chatbot

This project presents a fine-tuned, conversational AI-based chatbot designed for customer support in the banking and finance domain. The model acts as a customer service chatbot for a fictional bank.  It uses the `Zephyr-7B-Beta` large language model with LoRA (Low-Rank Adaptation) for efficient fine-tuning on custom instruction-response banking dataset.

## Features of this Chatbot

- Menu-based navigation for key banking service areas
- Free-text input support for real-world customer queries
- Context-aware responses (limited recent memory)
- Trained on a curated set of domain-specific FAQs and user queries compiled into a dataset
- Handles frustrated or emotionally charged inputs with empathy and answers dynamically

## How to Run

1. **Setup Google Colab or local environment with GPU**
2. **Install dependencies:**
3. **Run the codes in order:**
  python trainingbot.py
  python chatbotbank.py

# Requirements

Google Colab Pro recommended for GPU access

Python 3.8+

PyTorch with CUDA support (if using GPU)

Hugging Face Transformers

PEFT (Parameter-Efficient Fine-Tuning)
