import torch
import config as cf
from data_loader import MyDataLoader
from embeddings import PatchEmbedding


dataset = MyDataLoader(cf.DATA_DIR,cf.X_DIR,cf.Y_DIR)

train_set, test_set = torch.utils.data.random_split(dataset,[0.8,0.2])

sample_datapoint = torch.unsqueeze(train_set[0][0], 0)
print("Initial shape: ", sample_datapoint.shape)
#print(sample_datapoint)
embedding = PatchEmbedding()(sample_datapoint)
print("Embedded")
print("Embedding shape: ", embedding.shape)
#print(embedding)
