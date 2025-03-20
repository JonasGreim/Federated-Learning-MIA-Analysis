"""my-awesome-app: A Flower / PyTorch app."""
import glob
import os

import torch
import json
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters
from my_awesome_app.task import get_weights, load_data, set_weights, test, train, create_model
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor


class AttackerClient:
    def __init__(self, net, local_epochs):
        self.net = net
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)


def make_prediction():
    # Load untrained model
    device = "cpu"
    net = create_model()  # create model architecture/structure
    net.eval()
    net.to(device)

    # Load the latest model (weights/trained parameters)
    list_of_files = [fname for fname in glob.glob("../model_checkpoints/global_model_round_*")]
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from: ", latest_round_file)
    state_dict = torch.load(latest_round_file)  # Load the latest model (weights/trained parameters)
    net.load_state_dict(state_dict)

    # state_dict_ndarrays = [v.cpu().numpy() for v in net.state_dict().values()]
    # parameters = ndarrays_to_parameters(state_dict_ndarrays)

    # prepare image for prediction
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load and preprocess the image
    image_path = "./test_image.jpg"  # Change to your image path
    image = Image.open(image_path).convert("RGB")  # image.resize [32, 32]
    image = pytorch_transforms(image)  # image.shape [3, 32, 32]
    image = image.unsqueeze(0).to(device)  # Add batch dimension [1, 3, 32, 32]

    # make prediction
    with torch.no_grad():
        output = net(image)
    print("Output: ", output)
    predicted_class = torch.argmax(output, dim=1).item()
    print("Predicted class: ", predicted_class)
    # 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer, 5=dog, 6=frog, 7=horse, 8=ship, 9=truck


make_prediction()
