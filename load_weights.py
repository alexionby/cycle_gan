import torch
from models import InceptionGenerator


model = InceptionGenerator(image_size=128)
model.load_state_dict(torch.load("weights.pth"))

torch.save(model, "one_output.pt")