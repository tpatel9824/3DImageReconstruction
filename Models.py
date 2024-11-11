import sys


if __name__ == '__main__':
    from main import DDataset
    import torch
    import pandas as pd

    directory = r'C:\Users\91875\Downloads\pix3dorg'
    DataFrame = pd.read_csv(r'C:\Users\91875\Downloads\pix3dorg\Annotations.csv')
    Data = DDataset(DataFrame, directory)
    train_loader = torch.utils.data.DataLoader(Data, batch_size=10, num_workers=2, shuffle=True)

    # Ensure the code below is only executed after DataLoader is initialized
    train_features, train_labels = next(iter(train_loader))
    print("Feature Batch Shape", train_labels)
    for batch in train_loader:
        images,label=batch
        print("Size Of A Batch Is",images.shape)
        break


