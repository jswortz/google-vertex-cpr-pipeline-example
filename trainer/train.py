
import argparse
from urllib.request import urlretrieve
import torch
from torch.autograd import Variable
import pandas as pd
from google.cloud import storage

def train(args):

    DATA_DIR = "."

    LOCAL_DATA_FILE = f"{DATA_DIR}/iris.csv"

    urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        LOCAL_DATA_FILE,
    )

    ### Build a PyTorch NN Classifier
    print("PyTorch Version: {}".format(torch.__version__))


    CLASS_VOCAB = ["setosa", "versicolor", "virginica"]


    # Step 1. Load data
    # In this step, we are going to:

    # Load the data to Pandas Dataframe.
    # Convert the class feature (species) from string to a numeric indicator.
    # Split the Dataframe into input feature (xtrain) and target feature (ytrain).

    datatrain = pd.read_csv(
        LOCAL_DATA_FILE,
        names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
    )

    # change string value to numeric
    datatrain.loc[datatrain["species"] == "Iris-setosa", "species"] = 0
    datatrain.loc[datatrain["species"] == "Iris-versicolor", "species"] = 1
    datatrain.loc[datatrain["species"] == "Iris-virginica", "species"] = 2
    datatrain = datatrain.apply(pd.to_numeric)

    # change dataframe to array
    datatrain_array = datatrain.values

    # split x and y (feature and target)
    xtrain = datatrain_array[:, :4]
    ytrain = datatrain_array[:, 4]

    input_features = xtrain.shape[1]
    num_classes = len(CLASS_VOCAB)

    print("Records loaded: {}".format(len(xtrain)))
    print("Number of input features: {}".format(input_features))
    print("Number of classes: {}".format(num_classes))


    # Step 2. Set model parameters
    # You can try different values for hidden_units or learning_rate.

    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.1

    # Step 3. Define the PyTorch NN model
    # Here, we build a a neural network with one hidden layer, and a Softmax output layer for classification.

    model = torch.nn.Sequential(
        torch.nn.Linear(input_features, HIDDEN_UNITS),
        torch.nn.Sigmoid(),
        torch.nn.Linear(HIDDEN_UNITS, num_classes),
        torch.nn.Softmax(),
    )

    loss_metric = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Step 4. Train the model
    # We are going to train the model for num_epoch epochs.

    NUM_EPOCHS = 10000

    for epoch in range(NUM_EPOCHS):

        x = Variable(torch.Tensor(xtrain).float())
        y = Variable(torch.Tensor(ytrain).long())
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_metric(y_pred, y)
        loss.backward()
        optimizer.step()
        if (epoch) % 1000 == 0:
            print(
                "Epoch [{}/{}] Loss: {}".format(
                    epoch + 1, NUM_EPOCHS, round(loss.item(), 3)
                )
            )

    print("Epoch [{}/{}] Loss: {}".format(epoch + 1, NUM_EPOCHS, round(loss.item(), 3)))
    
    # Save the model to GCS

    storage_client = storage.Client(args.model_bucket)
    bucket = storage_client.bucket(args.model_bucket)
    blob = bucket.blob(args.model_blob)
    with blob.open("wb", ignore_flush=True) as f:
        torch.save(model, f)
        
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch Iris model')
    parser.add_argument('--project_id', type=str, help='GCP Project ID')
    parser.add_argument('--model_bucket', type=str, help='Model Bucket Name')
    parser.add_argument('--model_blob', type=str, help='Model Blob Path')
    
    args = parser.parse_args()
    
    train(args) #will return a torch model artifact
