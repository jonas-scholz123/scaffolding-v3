import torch
from torch.utils.data.dataloader import DataLoader

from scaffolding_v3.model.classification import ClassificationModule


def evaluate(
    model: ClassificationModule,
    val_loader: DataLoader,
    dry_run: bool = False,
) -> dict[str, float]:
    model.eval()
    val_loss = 0
    correct = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model.predict(data)
            val_loss += model.loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            if dry_run:
                break

    val_len = len(val_loader.dataset)  # type: ignore
    val_loss /= val_len
    return {"val_loss": val_loss, "val_accuracy": correct / val_len}
