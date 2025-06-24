import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader


def evaluate(
    model: nn.Module,
    loss_fn: nn.Module,
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
            output = model(data)
            val_loss += loss_fn(output, target).item()
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if dry_run:
                break

    val_len = len(val_loader.dataset)  # type: ignore
    val_loss /= val_len
    return {"val_loss": val_loss, "val_accuracy": correct / val_len}
