# Fake News Detection System using BERT and BiLSTM

This Django-based **Fake News Detection System** uses a **BERT-BiLSTM** hybrid model for analyzing news articles and headlines. It integrates **Gemini AI** for fact-checking and the **Google Fact-Checking API** for a daily email summary of detected fake news.

## Features
- **Hybrid Model**: Combines BERT and BiLSTM for contextual predictions.
- **Django Integration**: Fully functional Django app with templates for user interaction.
- **Fact-Checking**: Uses Gemini AI and Google Fact-Checking API for article verification.
- **Daily Email Summary**: Sends a daily email report of detected fake news using Google Fact-Checking API.

## Installation Steps

1. **Clone Repository**:
   ```bash
   git clone https://github.com/lbthinesh/Fake-News-Detection-System.git
   cd Fake-News-Detection-System
   ```

2. **Set Up Environment**:
   - Install dependencies with Pipenv:
     ```bash
     pipenv install
     ```

3. **Set Up Environment Variables**:
   - Create a `.env` file in `FakeNewsDetection/` to securely store your API keys and email credentials:
     
   - Add the following details to the `.env` file:
     ```env
     # Google Fact-Checking API Key
     GOOGLE_FACT_CHECK_API_KEY=your_google_fact_check_api_key

     # Gemini AI API Key
     GEMINI_API_KEY=your_gemini_api_key

     # Email Credentials
     EMAIL_HOST_USER=your_email@example.com
     EMAIL_HOST_PASSWORD=your_email_password
     ```
   - Update `settings.py` to load these environment variables. First, install `python-dotenv` if needed:
     ```bash
     pip install python-dotenv
     ```
   - In `FakeNewsDetection/settings.py`, add:
     ```python
     import os
     from dotenv import load_dotenv

     load_dotenv()  # Load environment variables from .env file

     GOOGLE_FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
     GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
     EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER")
     EMAIL_HOST_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD")
     ```

4. **Database Migration**:
   - Run migrations to set up the database:
     ```bash
     python manage.py migrate
     ```

5. **Run the Server**:
   ```bash
   python manage.py runserver

## File Structure

- **Main Django Project**: Located in `FakeNewsDetection/`
- **App Modules**: Located in `qa/`
- **Model Training**: Uses `model training/bert-bilstm-isot(f)2.ipynb`
- **Trained Model File**: Stored in `qa/model/best_model_isot.pth`

