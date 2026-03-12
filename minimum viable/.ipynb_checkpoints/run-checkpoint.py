import torch
import numpy as np


class SimpleCNN2(torch.nn.Module):
    def __init__(self, num_classes=14):
        super(SimpleCNN2, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=5)
        self.conv3 = torch.nn.Conv2d(16, 16, kernel_size=5)
        self.conv4 = torch.nn.Conv2d(16, 32, kernel_size=5)

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc1 = torch.nn.Linear(32, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.maxpool(torch.relu(self.conv1(x)))
        x = self.maxpool(torch.relu(self.conv2(x)))
        x = self.maxpool(torch.relu(self.conv3(x)))
        x = self.maxpool(torch.relu(self.conv4(x)))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = SimpleCNN2(12).to(device)
weights_path = 'training/ablation_study_2/ablation_original/model_best.pth'

model.load_state_dict(torch.load(weights_path, map_location=device))


# images are stored in /home/aric/vibrate_reduce/minimum viable/misalignment_x_combined_events_0000.npy where the last index ranges from 0000 to 0025
paths = [f'/home/aric/vibrate_reduce/minimum viable/misalignment_x_combined_events_{i:04d}.npy' for i in range(26)]

samples = []

for path in paths:
    sample_npy = np.load(path)[:, 130:570]
    max_val = np.max(sample_npy)
    if max_val == 0:
        img_normalized = sample_npy.astype(np.float32)
    else:
        img_normalized = sample_npy.astype(np.float32) / max_val

    img_tensor = torch.tensor(img_normalized, dtype=torch.float32)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    samples.append(img_tensor)

import time
proc_time = []
for i, sample in enumerate(samples):

    start_time = time.time()
    sample = sample.to(device)
    pred = model(sample)
    end_time = time.time()
    if i == 0:
        continue
    proc_time.append(1000 * (end_time - start_time))

print(f"Average inference time over {len(samples)} samples: {np.mean(proc_time):.2f} ms ± {np.std(proc_time):.2f} ms")
