---
title: Chatbot Mistral
emoji: 😷
colorFrom: white
colorTo: indigo
sdk: streamlit
sdk_version: 1.42.0
app_file: app.py
pinned: false
---

# Symptom Diagnosis Chatbot

An interactive Python chatbot that helps users identify a possible diagnosis based on the symptoms they report. This project integrates a knowledge base stored in a `database.txt` file and utilizes HuggingFace models for generating responses and performing translations.

## Description

This project was created to:
- **Load a Knowledge Base:** A `database.txt` file contains symptoms and their corresponding content. Each record follows the format:


- **Generate Customized Responses:** Using the `Mistral-7B-Instruct-v0.3` model from HuggingFace to interpret the user's query and the `Helsinki-NLP/opus-mt-tc-big-en-pt` model to translate responses into Portuguese.
- **Provide an Interactive Web Interface:** Built with Streamlit, allowing users to interact with the chatbot and adjust parameters via a sidebar.

## Features

- **Knowledge Base Integration:** Reads and formats symptoms and content from an external file.
- **Response Generation:** Provides responses based on the conversation context and the knowledge base.
- **Automatic Translation:** Translates responses to Portuguese.
- **Interactive Web Interface:** Developed with Streamlit, enabling a simple and intuitive conversation with the chatbot.
- **Customizable Settings:** Allows adjustments to system messages, maximum response length, and other parameters through the interface.

## Prerequisites

- **Python 3.8 or higher**
- **HuggingFace Access Token:** Set the environment variable `HF_TOKEN` with your HuggingFace token.
- The following Python libraries:
- `streamlit`
- `transformers`
- `langchain-huggingface`
- (Other dependencies may be listed in a `requirements.txt` file)

## Installation

1. **Clone the Repository**

2. **Install requirements**
```bash
pip install -r requirements.txt
```

**Highly recommend use hugginface spaces**