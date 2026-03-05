# EBOOK_RAG_System
Structure-aware RAG system with hierarchical metadata filtering and conversation memory.


An intelligent PDF book assistant that understands document structure - chapters, sections, and examples. Ask precise questions like "Explain Chapter 1 Example 1" and get accurate answers with page citations.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Features

- **📖 Structure-Aware Parsing**: Extracts TOC, chapters, sections, and examples using PyMuPDF
- **🔍 Hierarchical Retrieval**: Precise location-aware search (e.g., "Chapter 1 Example 1")
- **💬 Conversation Memory**: Maintains context across multiple questions
- **⚡ Fast Inference**: Powered by Groq (Llama 3.3 70B)
- **🎨 Clean UI**: Built with Streamlit for easy interaction

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- Git
- A [Groq API key](https://console.groq.com/keys) (free tier available)

 **SCREENSHOT**

<img width="1336" height="546" alt="image" src="https://github.com/user-attachments/assets/1f875a38-a984-4d23-9e9a-ea41032a231b" />

### Step 1: Clone the Repository

    
     git clone https://github.com/zaheertiwana544-stack/EBOOK_RAG_System.git
     cd EBOOK_RAG_System


### Step 2: Create Virtual Environment
Windows:
     python -m venv myenv
     myenv\Scripts\activate
Mac/Linux:

     python3 -m venv myenv
     source myenv/bin/activate


