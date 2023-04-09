import torch
import torchvision
import time
import os

# Download ResNet-152 from torchvision
resnet = torchvision.models.resnet152()
vgg = torchvision.models.vgg16()

# Optionally, you can load a pre-trained ResNet-152 model by uncommenting the following line
# resnet = torchvision.models.resnet152(pretrained=True)
# vgg = torchvision.models.vgg16(pretrained=True)


for i, model in enumerate([resnet, vgg]):
    # Set the model to evaluation mode
    model.eval()

    # Create a sample input tensor to test the model
    input_tensor = torch.randn(1, 3, 224, 224)

    # Script the model using torch.jit.script()
    scripted_model = torch.jit.script(model)

    # Test the model on the input tensor
    start = time.time()
    output = model(input_tensor)
    print("Time taken by model(): {:.4f} seconds".format(time.time() - start))

    # Get the top-5 predicted classes and their probabilities for the original model
    probabilities, indices = torch.topk(output, k=5)

    scripted_start = time.time()
    scripted_output = scripted_model(input_tensor)
    print("Time taken by scripted_model(): {:.4f} seconds".format(time.time() - scripted_start))

    # Get the top-5 predicted classes and their probabilities for the scripted model
    scripted_probabilities, scripted_indices = torch.topk(scripted_output, k=5)

    # Confirm that by compiling our model using torch.jit.script did not change the output of our model
    assert torch.allclose(indices, scripted_indices)

    # Print the output tensor's shape
    print(output.shape)

    # Define the path to the model file in the model_store directory
    model_file = os.path.join(os.path.dirname(__file__), "..", "model_store", "deployable" + str(i) + ".pt")

    # Save the scripted model to the model file
    scripted_model.save(model_file)

