import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

class HierarchicalPatchEmbedding(nn.Module):
    def __init__(self, image_size=512, num_levels=2, embedding_dim=48):
        super(HierarchicalPatchEmbedding, self).__init__()
        self.image_size = image_size
        self.num_levels = num_levels  # Number of levels in the hierarchy
        self.embedding_dim = embedding_dim
        assert embedding_dim % 3 == 0, "Embedding dimension must be divisible by 3."
        self.emb_dim_per_position = embedding_dim // 3  # Divide embedding among l, y, x

    def forward(self, image):
        """
        image: Tensor of shape (C, H, W)
        Returns:
            patches: List of tensors representing the image patches
            embeddings: Tensor of shape (num_patches, embedding_dim)
        """
        positions = self.generate_positions()
        patches = self.extract_patches(image, positions)
        embeddings = self.compute_embeddings(positions)
        return patches, embeddings  # patches: List of tensors, embeddings: Tensor

    def generate_positions(self):
        positions = []
        for l in range(self.num_levels + 1):
            num_patches_per_axis = 2 ** l  # Patches per axis at level l
            for y in range(num_patches_per_axis):
                for x in range(num_patches_per_axis):
                    positions.append((l, y, x))
        return positions  # List of tuples (l, y, x)

    def extract_patches(self, image, positions):
        """
        image: Tensor of shape (C, H, W)
        positions: List of tuples (l, y, x)
        Returns:
            patches: List of tensors
        """
        patches = []
        C, H, W = image.shape
        for pos in positions:
            l, y, x = pos
            num_patches_per_axis = 2 ** l
            patch_size = H // num_patches_per_axis  # Assuming square image
            y_start = y * patch_size
            y_end = (y + 1) * patch_size
            x_start = x * patch_size
            x_end = (x + 1) * patch_size
            patch = image[:, y_start:y_end, x_start:x_end]
            patches.append(patch)
        return patches  # List of tensors

    def compute_embeddings(self, positions):
        positions_tensor = torch.tensor(positions, dtype=torch.float32)  # Shape: (num_patches, 3)
        embeddings = self.get_sinusoid_encoding(positions_tensor)
        return embeddings  # Shape: (num_patches, embedding_dim)

    def get_sinusoid_encoding(self, positions):
        num_patches = positions.size(0)
        pe = torch.zeros(num_patches, self.embedding_dim)
        div_term = torch.exp(
            torch.arange(0, self.emb_dim_per_position, 2, dtype=torch.float32) *
            -(torch.log(torch.tensor(10000.0)) / self.emb_dim_per_position)
        )

        for i in range(3):  # For l, y, x
            pos = positions[:, i].unsqueeze(1)  # Shape: (num_patches, 1)
            pe_part = torch.zeros(num_patches, self.emb_dim_per_position)
            pe_part[:, 0::2] = torch.sin(pos * div_term)
            pe_part[:, 1::2] = torch.cos(pos * div_term)
            # Assign to the corresponding part of the embedding
            start = i * self.emb_dim_per_position
            end = (i + 1) * self.emb_dim_per_position
            pe[:, start:end] = pe_part
        return pe

class ConvModelWithSPP(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes):
        super(ConvModelWithSPP, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool_sizes = pool_sizes  # List of pooling sizes for SPP

    def forward(self, x):
        x = self.conv_layers(x)
        spp_features = self.spatial_pyramid_pool(x, self.pool_sizes)
        return spp_features

    def spatial_pyramid_pool(self, x, pool_sizes):
        batch_size, c, h, w = x.size()
        output = []
        for pool_size in pool_sizes:
            kernel_size = (h // pool_size, w // pool_size)
            stride = (h // pool_size, w // pool_size)
            pooling = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0)
            pooled = pooling(x).view(batch_size, -1)
            output.append(pooled)
        return torch.cat(output, dim=1)  # Shape: (batch_size, spp_output_length)

class HierarchicalSPPModel(nn.Module):
    def __init__(self, num_classes=4, image_size=512, num_levels=2, embedding_dim=48):
        super(HierarchicalSPPModel, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_levels = num_levels
        self.embedding_dim = embedding_dim
        self.patch_embedding = HierarchicalPatchEmbedding(
            image_size=image_size,
            num_levels=num_levels,
            embedding_dim=embedding_dim
        )
        self.out_channels = 128  # Number of output channels from the conv model
        self.pool_sizes = [1, 2, 4]  # Pooling sizes for SPP
        self.conv_model_spp = ConvModelWithSPP(
            in_channels=3,
            out_channels=self.out_channels,
            pool_sizes=self.pool_sizes
        )
        # Calculate the length of the output feature vector from SPP
        self.spp_output_length = sum([self.out_channels * (size ** 2) for size in self.pool_sizes])
        # Linear layer to project positional embeddings to match SPP output length
        self.projection_layer = nn.Linear(embedding_dim, self.spp_output_length)
        # Transformer encoder layer for self-attention
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.spp_output_length,
            nhead=8
        )
        # Final linear layer for classification
        self.classifier = nn.Linear(self.spp_output_length, num_classes)

    def forward(self, image):
        """
        image: Tensor of shape (batch_size, C, H, W)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        batch_size = image.size(0)
        # Initialize lists to store features from all images in the batch
        all_feature_vectors = []
        for i in range(batch_size):
            img = image[i]
            # Generate patches and positional embeddings
            patches, embeddings = self.patch_embedding(img)
            num_patches = len(patches)
            # Process patches through the convolutional model with SPP
            patch_features = []
            for j, patch in enumerate(patches):
                patch = patch.unsqueeze(0)  # Add batch dimension
                feature_vector = self.conv_model_spp(patch)  # Shape: (1, spp_output_length)
                # Project and add positional embeddings
                embedding = embeddings[j]  # Shape: (embedding_dim,)
                projected_embedding = self.projection_layer(embedding)  # Shape: (spp_output_length,)
                feature_vector = feature_vector + projected_embedding.unsqueeze(0)
                patch_features.append(feature_vector)
            # Stack feature vectors
            patch_features = torch.cat(patch_features, dim=0)  # Shape: (num_patches, spp_output_length)
            # Apply self-attention
            # Reshape to (sequence_length, batch_size=1, d_model)
            patch_features = patch_features.unsqueeze(1)
            attended_features = self.transformer_layer(patch_features)
            # Aggregate features (e.g., by averaging)
            aggregated_feature = attended_features.mean(dim=0).squeeze(0)  # Shape: (spp_output_length,)
            all_feature_vectors.append(aggregated_feature)
        # Stack features from all images in the batch
        all_feature_vectors = torch.stack(all_feature_vectors, dim=0)  # Shape: (batch_size, spp_output_length)
        # Pass through classifier
        logits = self.classifier(all_feature_vectors)  # Shape: (batch_size, num_classes)
        return logits

# Example usage:
def main():



    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize images to match your model's input size
        transforms.ToTensor(),
        # Add any additional transformations here
    ])

    # Load the training and test datasets
    train_dataset = datasets.STL10(root='./data', split='train',
                                            download=True, transform=transform)
    test_dataset = datasets.STL10(root='./data', split='test',
                                            download=True, transform=transform)

    # Define the classes you want to use
    classes_to_use = ['airplane', 'bird', 'car', 'cat']
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(train_dataset.classes)}
    classes_indices = [class_to_idx[class_name] for class_name in classes_to_use]

    # Filter the datasets
    def filter_by_class(dataset, classes_indices):
        indices = [i for i, (_, label) in enumerate(dataset) if label in classes_indices]
        return Subset(dataset, indices)

    train_dataset_filtered = filter_by_class(train_dataset, classes_indices)
    test_dataset_filtered = filter_by_class(test_dataset, classes_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset_filtered, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset_filtered, batch_size=64, shuffle=False)


    # Initialize the model
    num_classes = 4  # Adjust based on your selected classes
    model = HierarchicalSPPModel(num_classes=num_classes, image_size=512, num_levels=2, embedding_dim=48)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            # Move data to the appropriate device (CPU or GPU)
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

if __name__ == "__main__":
    main()
