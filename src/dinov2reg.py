import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
from torch.nn.functional import pad
import pytorch_lightning as pl

DATASET = "Soybean"
BATCH_SIZE = 1
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_EPOCHS = 100
id2label = {0: "Grain", 1: "Pod"}
num_classes = len(id2label)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Find the max width and height in this batch
    max_width = max(image.size(2) for image in images)
    max_height = max(image.size(1) for image in images)

    padded_images = []
    for image in images:
        padding = (0, max_width - image.size(2), 0, max_height - image.size(1))
        padded_image = pad(image, padding, mode='constant', value=0)
        padded_images.append(padded_image)

    images_tensor = torch.stack(padded_images)

    return {
        'pixel_values': images_tensor,
        'labels': labels
    }

TRAIN_DATASET = CocoDetection(root=f'/home/william/Projects/GrainPreHarvestDetection/data/{DATASET}/train', 
                            annFile=f'/home/william/Projects/GrainPreHarvestDetection/data/{DATASET}/train/coco.json', transform=transform)
VAL_DATASET = CocoDetection(root=f'/home/william/Projects/GrainPreHarvestDetection/data/{DATASET}/valid',
                            annFile=f'/home/william/Projects/GrainPreHarvestDetection/data/{DATASET}/valid/coco.json', transform=transform)
TEST_DATASET = CocoDetection(root=f'/home/william/Projects/GrainPreHarvestDetection/data/{DATASET}/test',
                            annFile=f'/home/william/Projects/GrainPreHarvestDetection/data/{DATASET}/test/coco.json', transform=transform)


print("Number of training examples:", len(TRAIN_DATASET))
print("Number of validation examples:", len(VAL_DATASET))
print("Number of test examples:", len(TEST_DATASET))

TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

class CustomDinoModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CustomDinoModel, self).__init__()
        self.base_model = base_model
        self.classification_head = nn.Linear(768, num_classes)  # Adjust input/output dimensions
        self.bbox_head = nn.Linear(768, 4)  # Bounding box predictions have 4 coordinates

    def forward(self, x):
        features = self.base_model(x)
        class_preds = self.classification_head(features)
        bbox_preds = self.bbox_head(features)
        return {'class': class_preds, 'bbox': bbox_preds}

class DinoLightningModule(pl.LightningModule):
    def __init__(self, model, num_classes):
        super(DinoLightningModule, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.classification_loss = nn.CrossEntropyLoss()  # For classification
        self.regression_loss = nn.SmoothL1Loss()  # For bounding box regression

    def forward(self, x):
        return self.model(x)

    def extract_targets(self, labels, device):
        class_targets = []
        bbox_targets = []
        for label in labels:
            if len(label) == 0:
                class_targets.append(torch.tensor([], dtype=torch.long).to(device))
                bbox_targets.append(torch.tensor([], dtype=torch.float).to(device))
            else:
                class_targets.append(torch.tensor([obj['category_id'] for obj in label], dtype=torch.long).to(device))
                bbox_targets.append(torch.tensor([obj['bbox'] for obj in label], dtype=torch.float).to(device))
        
        return class_targets, bbox_targets

    def training_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']

        outputs = self(pixel_values)

        class_preds, bbox_preds = outputs['class'], outputs['bbox']
        class_targets, bbox_targets = self.extract_targets(labels, self.device)

        total_loss = 0
        for i in range(len(class_targets)):
            if class_targets[i].numel() == 0:
                continue  # Skip if no targets

            class_loss = self.classification_loss(class_preds[i].unsqueeze(0), class_targets[i])
            bbox_loss = self.regression_loss(bbox_preds[i].unsqueeze(0), bbox_targets[i])

            total_loss += class_loss + bbox_loss

        total_loss /= len(class_targets)
        self.log('train_loss', total_loss,  prog_bar=True, batch_size=BATCH_SIZE)
        return total_loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']

        outputs = self(pixel_values)

        class_preds, bbox_preds = outputs['class'], outputs['bbox']
        class_targets, bbox_targets = self.extract_targets(labels, self.device)

        total_loss = 0
        for i in range(len(class_targets)):
            if class_targets[i].numel() == 0:
                continue  # Skip if no targets

            class_loss = self.classification_loss(class_preds[i].unsqueeze(0), class_targets[i])
            bbox_loss = self.regression_loss(bbox_preds[i].unsqueeze(0), bbox_targets[i])

            total_loss += class_loss + bbox_loss

        total_loss /= len(class_targets)
        self.log('val_loss', total_loss, prog_bar=True, batch_size=BATCH_SIZE)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer
    
if __name__ == "__main__":
    # Load the model
    base_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    model = CustomDinoModel(base_model, num_classes)
    dino_module = DinoLightningModule(model, num_classes)
    
    # Define the PyTorch Lightning trainer
    trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)  # Adjust parameters as needed
    
    # Train the model
    trainer.fit(dino_module, TRAIN_DATALOADER, VAL_DATALOADER)

    # Save the model
    torch.save(model.state_dict(), f"dinov2reg_{DATASET}.pth")
