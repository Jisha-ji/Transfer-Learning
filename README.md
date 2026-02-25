# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Develop an image classification model using transfer learning with the pre-trained VGG19 model.

## DESIGN STEPS
## STEP 1:
Import required libraries.Then dataset is loaded and define the training and testing dataset.

## STEP 2:
initialize the model,loss function,optimizer. CrossEntropyLoss for multi-class classification and Adam optimizer for efficient training.

## STEP 3:
Train the model with training dataset.

## STEP 4:
Evaluate the model with testing dataset.

## STEP 5:
Make Predictions on New Data.

## PROGRAM

```python
# Load Pretrained Model and Modify for Transfer Learning

model = models.vgg19(weights = models.VGG19_Weights.DEFAULT)

for param in model.parameters():
  param.requires_grad = False


# Modify the final fully connected layer to match the dataset classes

num_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_features,1)


# Include the Loss function and optimizer

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)



# Train the model

def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

    # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: JISHA BOSSNE SJ")
    print("Register Number: 2122224230106")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()



```


## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="947" height="768" alt="image" src="https://github.com/user-attachments/assets/240daf27-3796-4349-ad3c-57c0a9b626f5" />


### Confusion Matrix

<img width="802" height="605" alt="image" src="https://github.com/user-attachments/assets/8a4b4547-1cab-4ffb-9144-09940e018e3d" />


### Classification Report


<img width="587" height="213" alt="image" src="https://github.com/user-attachments/assets/e8651295-ede1-4adb-876a-22cf6b98f1df" />


### New Sample Prediction
<img width="557" height="812" alt="image" src="https://github.com/user-attachments/assets/b909354a-1b70-4caa-b7bc-ce139a962bc8" />


## RESULT
The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors.
