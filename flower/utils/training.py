import torch
from flower.utils.reproducibility import release_model


def train(net, trainloader, epochs, lr, lr_decay, weight_decay, device) -> tuple[float, float]:
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    print(f"weight_decay: {weight_decay}, lr: {lr}", flush=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    net.train()

    # Learning rate decay matching paper: decay = 1e-7
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: 1 / (1 + lr_decay * e)
    )

    running_loss = 0.0
    correct = 0
    total = 0

    for epoch in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)  # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # Get predicted class
            correct += (predicted == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Count total samples

        # Apply learning rate decay at end of each epoch
        scheduler.step()

    avg_trainloss = running_loss / (epochs * len(trainloader))
    avg_trainacc = correct / total  # Average accuracy over epochs

    release_model(net, device.type)
    return avg_trainloss, avg_trainacc


def test(net, testloader, device) -> tuple[float, float]:
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset) if len(testloader.dataset) > 0 else 0.0
    loss = loss / len(testloader)
    release_model(net, device.type)
    return loss, accuracy

