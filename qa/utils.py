from FakeNewsDetection.settings import BASE_DIR
import os
import google.generativeai as genai
from django.conf import settings
from django.core.mail import send_mail
import torch
from transformers import BertTokenizer


from torch import nn
from transformers import BertModel

class BERT_LSTM(nn.Module):

    def __init__(self, num_classes, hidden_size, num_layers, bidirectional):
        super(BERT_LSTM, self).__init__()
        # self.bert = BertModel.from_pretrained(os.path.join(BASE_DIR,'qa','model','bert-base-uncased'))
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze the last two layers of BERT
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True

        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(input_ids=sent_id, attention_mask=mask, return_dict=False, output_hidden_states=True)
        x = cls_hs[0]
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])
        return self.softmax(x)
    
model = BERT_LSTM(num_classes=2, hidden_size=128, num_layers=2, bidirectional=True)
model_path = os.path.join(BASE_DIR,'qa','model','best_model_isot.pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Use 'cpu' if you're on a CPU-only machine

# Set the model to evaluation mode
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



# Example usage of the model for prediction
def fake_news_detection(prompt):
    # Assuming tokenizer is defined and loaded (for example, from transformers)
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Make predictions
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        
    # Process output to get predicted class
    predicted_class = output.argmax(dim=1).item()

    if predicted_class==0:
        res="False"
    else:
        res="True"
    
    return res












from transformers import pipeline

def sentimentanalysis(prompt):
    # Load the pre-trained BERT sentiment analysis model
    sentiment_pipeline = pipeline("text-classification", model="textattack/bert-base-uncased-SST-2")

    # Perform sentiment analysis
    result = sentiment_pipeline(prompt)

    # Map sentiment labels to readable labels
    label_mapping = {
        "LABEL_0": "Negative sentiment",  # could be interpreted as a contradiction in your case
        "LABEL_1": "Positive sentiment",  # could be interpreted as support in your case
    }

    # Display the mapped label
    for res in result:
        label = label_mapping.get(res['label'], 'unknown')
        return label, str((round(res['score'],2))*100)














# Configure the Google Generative AI API key
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def get_google_ai_response(prompt):
    # Define the generation configuration
    generation_config = {
        "temperature": 1,                  # Controls the randomness of the output
        "top_p": 0.95,                     # Cumulative probability for nucleus sampling
        "top_k": 64,                       # Limits the number of tokens considered
        "max_output_tokens": 8192,        # Maximum number of tokens in the output
        "response_mime_type": "text/plain",  # Response format
    }

    prompt=prompt+" (answer the question only 'True' or 'False' or 'Nil' dont answer above one word)"
    # Create an instance of the GenerativeModel
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )

    # Start a chat session
    chat_session = model.start_chat(history=[])

    # Send the message and get the response
    response = chat_session.send_message(prompt)
    # return response.text

    if "true" in response.text.lower():
        return "True"
    elif "false" in response.text.lower():
        return "False"
    else:
        return None



def security_alert(user):
    subject="Your Journey to Combat Fake News Starts Here!"
    message="""
Dear {},

Thank you for signing up for Fake News Detection! We're excited to have you on board as we work together to fight misinformation and promote reliable information.

Here's what you can expect:

Real-Time Detection: Our system uses advanced algorithms to help identify fake news articles quickly and efficiently.
User-Friendly Interface: Easily navigate our platform to check articles and share insights with others.
Stay Informed: Receive updates and tips on how to spot fake news on your own.
To get started, simply log in to your account and explore the features we offer.

If you have any questions or need assistance, feel free to reach out to our support team at fakenewssupport@gmail.com.

Thank you for joining us in this important mission!

Best regards,
The Fake News Detection Team
            """.format(user)
    from_email=settings.EMAIL_HOST_USER
    to_email=[user.email]
    try:
        send_mail(subject, message, from_email, to_email)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")
