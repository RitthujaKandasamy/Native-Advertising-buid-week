
import torch 
from torch import nn, optim  
from model import model, Classifier, Classifier_3
from data_handler import train_loader, test_loader 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 




emb_dim = 300
MAX_SEQ_LEN = 64

# model = Classifier(MAX_SEQ_LEN, 300, 128)
model = Classifier(MAX_SEQ_LEN, 300, 128)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)



epochs = 15
all_train_losses, all_test_losses, all_accuracies = [],  [], []

for e in range(epochs):
     train_losses, test_losses, running_accuracy = 0, 0, 0

     for i, (sentences_train, labels_train) in enumerate(iter(train_loader)):

          sentences_train.resize_(sentences_train.size()[0], 64 * emb_dim)

          optimizer.zero_grad()
          prediction_train = model.forward(sentences_train)   
          loss_train = criterion(prediction_train, labels_train) 
          loss_train.backward()                  
          optimizer.step()                

          train_losses += loss_train.item()
     
     avg_train_loss = train_losses/len(train_loader)
     all_train_losses.append(avg_train_loss)


     model.eval()
     with torch.no_grad():
          for i, (sentences_test, labels_test) in enumerate(iter(test_loader)):
               sentences_test.resize_(sentences_test.size()[0], 64* emb_dim)

               prediction_test = model.forward(sentences_test) 
               loss_test = criterion(prediction_test, labels_test) 

               test_losses += loss_test.item()


               prediction_class = torch.argmax(prediction_test, dim=1)
               running_accuracy += accuracy_score(labels_test, prediction_class)
          
          avg_test_loss = test_losses/len(test_loader)
          all_test_losses.append(avg_test_loss)

          avg_running_accuracy = running_accuracy/len(test_loader)
          all_accuracies.append(avg_running_accuracy)


     model.train()


     print(f'Epoch  : {e+1:3}/{epochs}    |   Train Loss:  : {avg_train_loss:.8f}     |  Test Loss:  : {avg_test_loss:.8f}  |  Accuracy  :   {avg_running_accuracy:.4f}')

torch.save({ "model_state": model.state_dict(), 'max_seq_len' : 64, 'emb_dim' : 64, 'hidden1' : 32, 'hidden2' : 32}, 'new_trained_model_2')

plt.plot(all_train_losses, label='Train Loss')
plt.plot(all_test_losses,  label='Test Loss')
plt.plot(all_accuracies,   label='Accuracy')

plt.legend()
plt.show()


















































"""
model = model
# print(model)

emb_dim, max_seq_length  = 2, 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


epochs, all_train_losses, all_test_losses, all_accuracies = 3, [], [], []

for e in range(epochs):
     train_losses, test_losses, running_accuracy = 0, 0, 0

     for i, (sentences_train, labels_train) in enumerate(iter(train_loader)):
          # sentences_train.resize_(sentences_train.size()[0], max_seq_length * emb_dim)
          sentences_train.resize_(sentences_train.size()[0], max_seq_length * emb_dim).to(torch.long)

          optimizer.zero_grad()
          prediction_train = model.forward(sentences_train)   
          loss_train = criterion(prediction_train, labels_train) 
          loss_train.backward()   
          optimizer.step()

          train_losses += loss_train.item()

     avg_train_loss = train_losses/len(train_loader)
     all_train_losses.append(avg_train_loss)


     model.eval()
     with torch.no_grad():
          for i, (sentences_test, labels_test) in enumerate(iter(test_loader)):
               sentences_test.resize_(sentences_test.size()[0], max_seq_length * emb_dim)

               prediction_test = model.forward(sentences_test) 
            #    probability_test = torch.sigmoid(prediction_test)
               loss_test = criterion(prediction_test, labels_test) 

               test_losses += loss_test.item()

            #    classes = probability_test > 0.5

            #    running_accuracy += ((classes == labels_test).all(dim=1)).sum()/len(labels_test)

               prediction_class = torch.argmax(prediction_test, dim=1) 
               running_accuracy += accuracy_score(labels_test, prediction_class)
          

          avg_test_loss = test_losses/len(test_loader)
          all_test_losses.append(avg_test_loss)

          avg_running_accuracy = running_accuracy/len(test_loader)
          all_accuracies.append(avg_running_accuracy)


     model.train()


     print(f'Epoch  : {e+1:3}/{epochs}    |   Train Loss:  : {avg_train_loss:.8f}     |  Test Loss:  : {avg_test_loss:.8f}  |  Accuracy  :   {avg_running_accuracy:.4f}')

torch.save({ "model_state": model.state_dict(), 'max_seq_len' : 256}, 'TRAINED_MODEL')

plt.plot(all_train_losses, label='Train Loss')
plt.plot(all_test_losses,  label='Test Loss')
plt.plot(all_accuracies,   label='Accuracy')
plt.legend()
plt.show()


"""

