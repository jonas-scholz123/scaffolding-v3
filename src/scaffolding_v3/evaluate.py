import torch
from torch.utils.data.dataloader import DataLoader

from scaffolding_v3.model.classification import ClassificationModule


@torch.no_grad()
def evaluate(
    model: ClassificationModule,
    dataloader: DataLoader,
    dry_run: bool = False,
) -> dict[str, float]:
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    device = next(model.parameters()).device

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model.predict(data)
        val_loss += model.loss_fn(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if dry_run:
            break
    if total == 0:
        return {"loss": float("inf"), "_accuracy": 0.0}
    return {"loss": val_loss / total, "accuracy": correct / total}
