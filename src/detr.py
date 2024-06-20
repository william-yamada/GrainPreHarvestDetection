import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import cv2
import torch
import supervision as sv
import matplotlib.pyplot as plt
import os
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from coco_eval import CocoEvaluator
from tqdm import tqdm

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'facebook/detr-resnet-50'
SAVE_PATH = 'detr-soybean'
CONFIDENCE_TRESHOLD = 0.1
IOU_TRESHOLD = 0.8
MAX_EPOCHS = 100
BATCH_SIZE = 32
TRAIN = True
DATASET = 'Wheat'

# id2label = {0: "grain", 1: "husk", 2: "residue", 3: "spike"}
id2label = {0: "Grain", 1: "Pod"}

# Test image
IMAGE_PATH = '/home/william/Projects/GrainPreHarvestDetection/data/Wheat/valid/images/2384_jpeg.rf.6945f9b1297f3ff0d1bd1cb17aedcb57.jpg'

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self, 
        image_directory_path: str, 
        image_processor, 
        train: bool = True
    ):
        annotation_file_path = os.path.join(image_directory_path, "coco.json")
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)        
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target

def inference_from_checkpoint (checkpoint_path, image_path, confidence_treshold, iou_treshold, plot=False):
    model = DetrForObjectDetection.from_pretrained(checkpoint_path)
    image_processor = DetrImageProcessor.from_pretrained(checkpoint_path)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    outputs = model(image_processor(images=image, return_tensors="pt"))

    # keep only predictions with confidence > treshold
    keep = outputs["pred_logits"][0][:, -1] > confidence_treshold
    outputs["pred_logits"] = outputs["pred_logits"][0][keep]
    outputs["pred_boxes"] = outputs["pred_boxes"][0][keep]

    # non-maximum suppression
    keep = torchvision.ops.nms(outputs["pred_boxes"], outputs["pred_logits"], iou_treshold)
    outputs["pred_logits"] = outputs["pred_logits"][keep]
    outputs["pred_boxes"] = outputs["pred_boxes"][keep]

    if plot:
        plt.imshow(image)
        ax = plt.gca()
        for score, (x0, y0, x1, y1), label in zip(outputs["pred_logits"], outputs["pred_boxes"], outputs["pred_classes"]):
            box = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="red", linewidth=2)
            ax.add_patch(box)
            plt.text(x0, y0, f"{id2label[label.item()]}: {score.item():.2f}", color="white", verticalalignment="top")
        plt.axis("off")
        plt.show()

    return outputs

def collate_fn(batch):
    # DETR authors employ various image sizes during training, making it not possible 
    # to directly batch together images. Hence they pad the images to the biggest 
    # resolution in a given batch, and create a corresponding binary pixel_mask 
    # which indicates which pixels are real/which are padding
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

class Detr(pl.LightningModule):
     def __init__(self, lr, lr_backbone, weight_decay):
         super().__init__()
         # replace COCO classification head with custom head
         # we specify the "no_timm" variant here to not rely on the timm library
         # for the convolutional backbone
         self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                             revision="no_timm",
                                                             num_labels=len(id2label),
                                                             ignore_mismatched_sizes=True)
         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay

     def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs

     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss, prog_bar=True,batch_size=BATCH_SIZE)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item(), prog_bar=True,batch_size=BATCH_SIZE)

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, prog_bar=True,batch_size=BATCH_SIZE)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item(), prog_bar=True,batch_size=BATCH_SIZE)

        return loss

     def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)

        return optimizer

     def train_dataloader(self):
        return TRAIN_DATALOADER

     def val_dataloader(self):
        return VAL_DATALOADER

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

# Load the model
image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

TRAIN_DATASET = CocoDetection(
    image_directory_path=os.path.join(f"/home/william/Projects/GrainPreHarvestDetection/data/{DATASET}/train"),
    image_processor=image_processor, 
    train=True)
VAL_DATASET = CocoDetection(
    image_directory_path=os.path.join(f"/home/william/Projects/GrainPreHarvestDetection/data/{DATASET}/valid"), 
    image_processor=image_processor, 
    train=False)
TEST_DATASET = CocoDetection(
    image_directory_path=os.path.join(f"/home/william/Projects/GrainPreHarvestDetection/data/{DATASET}/test"), 
    image_processor=image_processor, 
    train=False)

print("Number of training examples:", len(TRAIN_DATASET))
print("Number of validation examples:", len(VAL_DATASET))
print("Number of test examples:", len(TEST_DATASET))

TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=BATCH_SIZE)
TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=BATCH_SIZE)

model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

if TRAIN:
    # Train the model
    trainer = Trainer(devices=1, accelerator="gpu", max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)

    trainer.fit(model)

    # Save the model
    model.model.save_pretrained("detr-soybean") # save the model

model = Detr.load_from_checkpoint("/home/william/Projects/GrainPreHarvestDetection/src/lightning_logs/version_1/checkpoints/epoch=99-step=300.ckpt",
                                  lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

model.to(DEVICE)

evaluator = CocoEvaluator(coco_gt=TEST_DATASET.coco, iou_types=["bbox"])
print("Running evaluation...")

for idx, batch in enumerate(tqdm(TEST_DATALOADER)):
    pixel_values = batch["pixel_values"].to(DEVICE)
    pixel_mask = batch["pixel_mask"].to(DEVICE)
    labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

    with torch.no_grad():
      outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = image_processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes,threshold=CONFIDENCE_TRESHOLD)

    predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
    predictions = prepare_for_coco_detection(predictions)
    evaluator.update(predictions)

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()