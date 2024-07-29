import time

import torch.optim
import torch.nn.functional as F
from dora_implementation.eval.evaluation import compute_accuracy


def train(num_epochs, model, optimizer, train_loader, device):

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.view(-1, 28 * 28).to(device)
            targets = targets.to(device)

            # FORWARD AND BACK PROP
            logits = model(features)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            if not batch_idx % 400:
                print(
                    "Epoch: %03d/%03d | Batch %03d/%03d | Loss: %.4f"
                    % (epoch + 1, num_epochs, batch_idx, len(train_loader), loss)
                )

        with torch.set_grad_enabled(False):
            print(
                "Epoch: %03d/%03d training accuracy: %.2f%%"
                % (epoch + 1, num_epochs, compute_accuracy(model, train_loader, device))
            )

        print("Time elapsed: %.2f min" % ((time.time() - start_time) / 60))

    print("Total Training Time: %.2f min" % ((time.time() - start_time) / 60))
