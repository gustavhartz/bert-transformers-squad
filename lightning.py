
import torch
import pytorch_lightning as pl
# simple pytorch training loop using tqdm to show progress


class PLQAModel(pl.LightningModule):
    def __init__(self, model, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.model = model

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']
        outputs = self.model(input_ids, attention_mask=attention_mask,
                             start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        self.log("train_loss", loss)
        return {'loss': loss, 'log': {'train_loss': loss}}

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
        return {'loss': loss, 'log': {'val_loss': loss}}

    def validation_end(self, outputs):
        ct, sum = 0, 0
        for pred in outputs:
            sum += pred['loss']
            ct += 1
        return sum / ct

    def train_end(self, outputs):
        ct, sum = 0, 0
        for pred in outputs:
            sum += pred['loss']
            ct += 1
        return sum / ct

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
