import wandb
import os
import numpy as np
from sklearn.metrics import confusion_matrix


def init_wandb(model, args=None) -> None:
    """
    Initialize project on Weights & Biases
    """
    wandb.login(key=args.wandb_api_key)
    wandb.init(
        name=args.wandb_name,
        project="traffic-sign",
        id=args.wandb_id,
        dir="./",
    )
    if args:
        wandb.config.update(args)

    wandb.watch(model, log="all")


def wandb_log(train_loss: float, val_loss: float, val_acc: float, epoch: int):
    """
    Logs the accuracy and loss to wandb
    """
    wandb.log({
        'Training loss': train_loss,
        'Validation loss': val_loss,
        'Validation Accuracy': val_acc
    })


def wandb_save_summary(test_acc: float):
    """
    Saves Test accuracy in wandb
    """

    wandb.run.summary["test_accuracy"] = test_acc


def wandb_log_conf_matrix(y_true: list, y_pred: list):
    """Logs the confusion matrix

    Args:
        y_true (list): ground truth labels
        y_pred (list): predicted labels
    """

    wandb.log({"conf_mat": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_true,
        preds=y_pred
    )})


def save_model_wandb(save_path):
    """ Saves model to wandb

    Args:
        save_path (str): Path to save the wandb model
    """

    wandb.save(os.path.abspath(save_path))
