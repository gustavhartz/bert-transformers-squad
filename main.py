
# Run training of the model using pytorch lightning
from lightning import PLQAModel
from models import QAModelBert
import pytorch_lightning as pl
import torch
from data import SquadDataset
from torch.utils.data import DataLoader


def main():
    hparams = {
        'lr': 0.001,
        'max_seq_length': 128,
        'batch_size': 8,
        'num_workers': 4,
        'num_labels': 2,
        'hidden_size': 768,
        'num_train_epochs': 3,
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

    trainer = pl.Trainer(gpus=1, max_epochs=hparams['num_train_epochs'])
    trainer.fit(model, train_loader, val_loader)

    torch.save(model.model, "model.model")


if __name__ == "__main__":
    main()
