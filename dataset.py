from torchvision import transforms

images_dir = "C:\Users\tpate\OneDrive\Documents\Deep learning\class assignmets"
labels_dir = "C:\Users\tpate\OneDrive\Documents\Deep learning\class assignmets\pix3d_full\model" 

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

dataset = CustomDataset(images_dir=images_dir, labels_dir=labels_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
