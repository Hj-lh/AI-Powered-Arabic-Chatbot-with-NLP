# AI-Powered Arabic Chatbot with NLP


This project is an AI-based chatbot designed to assist students by providing information about professors' office locations, exam times, and other related inquiries. The chatbot interacts in Arabic and is capable of managing context to provide accurate responses based on user queries.


![download](https://github.com/Hj-lh/NLP-basedu-rule-arabic-chatbot/assets/160587130/5b6a55dd-cb8b-4d82-b71a-bbdc19386756)
![image](https://github.com/Hj-lh/NLP-basedu-rule-arabic-chatbot/assets/160587130/8e77ddb7-ffbc-4259-ab7b-b7e913444f04)



## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)A
- [Installation](#installation)
- [Usage](#usage)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Context Management](#context-management)
- [License](#license)

## Project Overview

The Student Guide Chatbot is built to help students get quick answers to their academic-related questions. It leverages machine learning models and natural language processing to understand and respond to queries effectively.

## Features

- Provides information about professors' office locations
- Notifies students of exam schedules
- Manages context to maintain conversation flow
- Supports Arabic language

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/student-guide-chatbot.git
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Camel tools:
   ```bash
   pip install camel-tools
## Usage

1. Run the application:
    ```bash
    python app.py
    ```
2. Access the chatbot interface through your web browser at `http://localhost:5000`.

## Preprocessing

Preprocessing is a crucial step to prepare data for training the chatbot model. It involves cleaning and structuring the data to ensure the model receives high-quality input.

### Steps:

1. **Data Collection**: Gather conversational data relevant to students' academic inquiries.
2. **Text Cleaning**: Remove unnecessary characters, normalize text, and handle misspellings.
3. **Tokenization**: Split text into tokens (words or phrases) to feed into the model.
4. **Labeling**: Annotate the data with appropriate labels for supervised learning.

Scripts involved: `preprocess.py`

## Training

Training the model involves using the preprocessed data to teach the chatbot how to respond to various queries.

### Steps:

1. **Model Selection**: Choose an appropriate model architecture (e.g., Transformer, LSTM).
2. **Training Configuration**: Set hyperparameters such as learning rate, batch size, and epochs.
3. **Model Training**: Feed the preprocessed data into the model and train it.
4. **Evaluation**: Assess the model's performance using a validation dataset and fine-tune as necessary.

Scripts involved: `train.py`

## Context Management

Effective context management ensures the chatbot maintains the flow of conversation, providing relevant responses based on previous interactions.

### Steps:

1. **State Tracking**: Keep track of user interactions and the current state of the conversation.
2. **Context Storage**: Store relevant information such as user queries and responses.
3. **Context Retrieval**: Retrieve stored context to inform subsequent responses.

Scripts involved: `context_manager.py`



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
