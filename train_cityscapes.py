import torch
from dataset.cityscapes import CityscapeDataModule
from utils import build_student_model, compute_loss
from tqdm import tqdm
import os

def train_cityscapes(model, dataloader, optimizer, scheduler, device):
    model.to(device)
    model.train()
    
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", unit="batch"):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        student_logits = model(images)
        
        loss_dict = compute_loss(student_logits, labels)
        loss = loss_dict['total']
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
    
    

def validate_cityscapes(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_pixels = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", unit="batch"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            student_logits = model(images)
            
            _, predicted = torch.max(student_logits, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_pixels += labels.numel()
    
    return total_correct / total_pixels
    
    
def main(encoder='timm-efficientnet-b0', decoder='unet', weight=None, dropout=0.1):
    root = '/media/esr/ssd0/cityscapes'
    dm = CityscapeDataModule(root, batch_size=4, num_workers=2, size=(256, 256))
    dm.setup()
    
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_student_model(encoder=encoder, decoder=decoder, weight=weight, dropout=dropout)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)
    
    num_epochs = 20
    print("Starting training...")
    for epoch in range(num_epochs):
        loss = train_cityscapes(model, train_loader, optimizer, scheduler, device)
        acc = validate_cityscapes(model, val_loader, device)
        print(f"Validation Accuracy: {acc:.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")

    print("Training complete.")
    
    # Save the trained model
    save_path = 'saved_models/cityscapes'
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), f'{save_path}/{encoder}_{decoder}_dropout-{dropout}.pth')

    
if __name__ == "__main__":
    main(encoder='timm-efficientnet-b0', decoder='unet', weight='imagenet', dropout=0.5)

