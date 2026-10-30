import warning_filters  # Configure warnings before importing torch.
import torch.nn as nn
from tqdm import tqdm
from utils import MetricsTop
import torch


def do_test(args, model, dataloader):
    criterion = nn.L1Loss()
    metrics = MetricsTop().getMetrics(args.dataset_name)
    model.eval()
    y_pred, y_true = [], []
    eval_loss = 0.0

    with torch.no_grad():
        with tqdm(dataloader, desc="Evaluating", leave=False) as td:
            for batch_data in td:
                vision = batch_data['vision'].to(args.device)
                audio = batch_data['audio'].to(args.device)
                text = batch_data['text'].to(args.device)
                labels = batch_data['labels']['M'].to(args.device)

                labels = labels.view(-1, 1)

                outputs = model(text, audio, vision)['M']

                # compute loss
                loss = criterion(outputs, labels)
                eval_loss += loss.item()

                y_pred.append(outputs.cpu())
                y_true.append(labels.cpu())

            eval_loss = eval_loss / len(dataloader)
            pred, true = torch.cat(y_pred), torch.cat(y_true)

    eval_results = metrics(pred, true)
    eval_results["Loss"] = round(eval_loss, 4)
    return eval_results
