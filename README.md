# SimplePythonChatbot
A simple AI chatbot in python that talks about video games. 

This program makes use of the following libraries:
- nltk_utils
- numpy
- torch

# Process Explained
- First a JSON intents list is created. This is what possible responses to questions could be and what is used to train the model
- Then the model is created where all it's attributes are defined
- In order to use the neurel net model it must be trained on the intents data
- When model is being trained it is ideal to have an accurracy above 80%
- In the **chat.py** this where the program comes togther: the training data is used to generate a chatbot that can respond to your questions about gaming

# Further Improvements
- Chatbot is currently not well versed in any other topics... add functionality to learn from each chat session


# Learned Outcomes
- How to create a proper training set using a Bag Of Words method
