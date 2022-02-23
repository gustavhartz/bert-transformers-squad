
# Run training of the model using pytorch lightning
from lightning import PLQAModel
from models import QAModelBert
import pytorch_lightning as pl
import torch
from data import SquadDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from transformers import BertTokenizerFast

# Not tested
class LogTextSamplesCallback(Callback):
    def on_training_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            """Called when the validation batch ends."""

            # `outputs` comes from `LightningModule.validation_step`
            # which corresponds to our model predictions in this case

            # Let's log 20 sample image predictions from first batch
            if batch_idx == 0:
                wandb_logger=trainer.logger
                tokenizer = pl_module.tokenizer
                questions = [tokenizer.decode(x['input_ids'][x['token_type_ids'].nonzero().squeeze()]) for x in batch]
                answers = [tokenizer.decode(x['input_ids'][x['start_positions']:x['end_positions']]) for x in batch]
                start_pred = torch.argmax(outputs[1], dim=1)
                end_pred = torch.argmax(outputs[2], dim=1)
                preds = [tokenizer.decode(x[2][x[0]:x[1]]) if x[0]<x[1] else "fes" for x in zip(start_pred,end_pred, batch['input_ids']) ]
                # Option 2: log predictions as a Table
                columns = ['question', 'answer', 'prediction']
                data = [questions,answers,preds]
                wandb_logger.log_text(key='traning_predicition_sample', columns=columns, data=data)

def main():
    global hparams
    hparams = {
        'lr': 5e-5,
        'batch_size': 16,
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
    
    wandb_logger = WandbLogger(project="bert-squad", entity="gustavhartz")
    
    print("Batch size", hparams.get("batch_size"))
    train_loader = DataLoader(
        train_dataset, batch_size=hparams.get("batch_size"), shuffle=True, num_workers=hparams.get('num_workers'))
    val_loader = DataLoader(
        val_dataset, batch_size=hparams.get("batch_size"), shuffle=True, num_workers=hparams.get('num_workers'))
    
    del train_encodings
    del val_encodings
    model = QAModelBert(hparams, hparams['bert_model'])
    litModel = PLQAModel(model, hparams)    
    trainer = pl.Trainer(gpus=2, max_epochs=hparams['num_train_epochs'], logger=wandb_logger, strategy='dp')
    trainer.fit(litModel, train_loader, val_loader)
    torch.save(model.model, "model.model")


if __name__ == "__main__":
    main()
