
import torch
import pytorch_lightning as pl
# simple pytorch training loop using tqdm to show progress


class PLQAModel(pl.LightningModule):
    def __init__(self, model, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.model = model
        
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        batch=x
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']
        return outputs

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']
        outputs = self.model(input_ids, attention_mask=attention_mask,
                             start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']
        outputs = self.model(input_ids, attention_mask=attention_mask,
                             start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        self.log(
            "val_loss",
            loss,
            on_epoch=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
