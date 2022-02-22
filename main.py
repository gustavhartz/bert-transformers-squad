
# Run training of the model using pytorch lightning
from lightning import PLQAModel
from models import QAModelBert
import pytorch_lightning as pl
import torch
from data import SquadDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger


def main():
    hparams = {
        'lr': 5e-5,
        'batch_size': 14,
        'num_workers': 4,
        'num_labels': 2,
        'hidden_size': 768,
        'num_train_epochs': 4,
        'bert_model': 'bert-base-uncased',
    }
    train_encodings = torch.load("./squad/train_encodings")
    val_encodings = torch.load("./squad/val_encodings")
    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)
    train_loader = DataLoader(
        train_dataset, batch_size=hparams.get("batch_size"), shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=hparams.get("batch_size"), shuffle=True)

    del train_encodings
    del val_encodings
    model = QAModelBert(hparams, hparams['bert_model'])
    litModel = PLQAModel(model, hparams)
    # Logger
    wandb_logger = WandbLogger(project="bert-squad", entity="gustavhartz")
    
    trainer = pl.Trainer(gpus=2, max_epochs=hparams['num_train_epochs'], logger=wandb_logger, strategy='dp')
    trainer.fit(litModel, train_loader, val_loader)
    torch.save(model.model, "model.model")


if __name__ == "__main__":
    main()
