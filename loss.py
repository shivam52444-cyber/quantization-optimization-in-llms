from dataset_loader import *
from loader import *
from texttoken import *

outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss

print("Loss:", loss.item())