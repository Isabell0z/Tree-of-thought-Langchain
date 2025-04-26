# 🧠 Tree of Thought for Game 24 using LangChain

This project implements a **Tree of Thought (ToT)** reasoning agent for solving the classic **Game 24** puzzle, built with the [LangChain](https://www.langchain.com/) framework.

> **Game 24** is a math puzzle: given 4 numbers (1–9), use +, −, ×, ÷ to make the number 24.  
> This system uses language models to explore reasoning trees and find valid solutions.

---

## 🚀 Features

- 🌳 Tree of Thought agent using LangChain
- 🧠 Custom prompt templates for mathematical reasoning
- 🔎 Supports BFS style ToT solvers
- ✅ Checker module to verify correct final answers (e.g., equals 24)
- 📈 Thought pruning, scoring, and configurable search depth

---
## 🛠️ Project Structure
```
├── README.txt
├── __pycache__
├── checker.py             # Thought evaluator
├── game24.csv             # Dataset
├── main.py                # Run
├── promptStrategy.py      # Prompt templates for thought generation
├── thought.py
├── totChain.py
└── totMemory.py
```

