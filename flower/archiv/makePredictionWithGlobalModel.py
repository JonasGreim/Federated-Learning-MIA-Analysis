# """my-awesome-app: A Flower / PyTorch app."""
# import glob
# import os
#
# import torch
# from flower.task import get_weights, load_data, set_weights, test, train, create_model
# from PIL import Image
# from torchvision.transforms import Compose, Normalize, ToTensor
#
#
# def make_prediction():
#     # Load untrained model
#     device = "cpu"
#     net = create_model(model_name="mia_paper")  # create model architecture/structure
#     net.eval()
#     net.to(device)
#
#     # Load the latest model (weights/trained parameters)
#     list_of_files = [fname for fname in glob.glob("../model_checkpoints/global_model_round_*")]
#     latest_round_file = max(list_of_files, key=os.path.getctime)
#     print("Loading pre-trained model from: ", latest_round_file)
#     state_dict = torch.load(latest_round_file)  # Load the latest model (weights/trained parameters)
#     net.load_state_dict(state_dict)
#
#     # state_dict_ndarrays = [v.cpu().numpy() for v in net.state_dict().values()]
#     # parameters = ndarrays_to_parameters(state_dict_ndarrays)
#
#     # prepare image for prediction
#     pytorch_transforms = Compose(
#         [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )
#
#     # Load and preprocess the image
#     image_path = "../../images/test_image.jpg"  # Change to your image path
#     image = Image.open(image_path).convert("RGB")  # image.resize [32, 32]
#     image = pytorch_transforms(image)  # image.shape [3, 32, 32]
#     image = image.unsqueeze(0).to(device)  # Add batch dimension [1, 3, 32, 32]
#
#     # make prediction/inference
#     with torch.no_grad():
#         output = net(image)
#     # print("Output: ", output)
#     predicted_class = torch.argmax(output, dim=1).item()  # confidence score for classes: tensor([[-2.3889, -1.0582, -0.1234, -0.5264, -1.0304, -0.6134, -2.9557,  0.1385, -2.1263, -1.3092]])
#     # before applying an activation function like softmax to convert them into probabilities
#     print("Predicted class number: ", predicted_class)
#     # 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer, 5=dog, 6=frog, 7=horse, 8=ship, 9=truck
#     class_mapping = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
#     print("Predicted class: ", class_mapping[predicted_class])
#
#     output_probabilities = torch.nn.functional.softmax(output, dim=1)
#     # print("Output probabilities: ", output_probabilities)
#
#     class_probabilities = {class_mapping[i]: round(prob.item(), 2) for i, prob in enumerate(output_probabilities[0])}
#     print("Class probabilities: ", class_probabilities)
#     # Output probabilities:  tensor([[0.0546, 0.1229, 0.5366, 0.0244, 0.0407, 0.0259, 0.0347, 0.0039, 0.1158, 0.0405]])
#
#
# make_prediction()
