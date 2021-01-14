import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # check to see if gpu is available for use

with open('intents.json', 'r') as f:
    intents = json.load(f)

# load pre-processed data file
FILE = "data.pth"
data = torch.load(FILE)

# get all data from save file
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
input_size = data["input_size"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Chief"
print("Let's chat! type 'quit' to exit")
while True:
    sentence = input('You: ')
    if sentence == 'quit':
        break
    # tokenize sentence to be processed for a response
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)  # put the user sentence in the model to be used to predict a response
    _, predicted = torch.max(output, dim=1)  # predict a response
    tag = tags[predicted.item()]  # predicted tag

    # check if probabilty of predicted tag is high enough to warrant a correct response
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
    # loop over all intents and check if the predicted tag matches
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f'{bot_name}: {random.choice(intent["responses"])}')  # find a random response to display
    else:
        print(f"{bot_name}: I do not understand...")
