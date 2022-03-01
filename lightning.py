
import torch
import pytorch_lightning as pl
from utils import calculate_batch_accuracy
# simple pytorch training loop using tqdm to show progress


class PLQAModel(pl.LightningModule):
    def __init__(self, model, hparams, tokenizer):
        super().__init__()
        self.hparams.update(hparams)
        self.model = model
        self.save_hyperparameters()
        self.tokenizer = tokenizer

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        batch = x
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']
        outputs = self.model(input_ids, attention_mask=attention_mask,
                             start_positions=start_positions, end_positions=end_positions)
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

        return {'loss': loss, 'pred': [outputs[1], outputs[2]]}

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
        )
        rs = calculate_batch_accuracy(
            batch, outputs[1], outputs[2], self.tokenizer)
        # Log text examples
        return {'loss': loss, 'pred': [outputs[1], outputs[2]], 'f1': rs.get('f1'), 'exact_match': rs.get('exact_match')}

    def training_epoch_end(self, outputs):
        ct, _sum = 0, 0
        for pred in outputs:
            _sum += pred['loss'].item()
            ct += 1
        self.log(
            "epoch_train_loss",
            _sum / ct,
            sync_dist=True
        )

    def validation_epoch_end(self, outputs):
        ct, _sum = 0, 0
        em_sum, f1_sum = 0, 0
        for pred in outputs:
            _sum += pred['loss'].item()
            em_sum += pred['exact_match']
            f1_sum += pred['f1']
            ct += 1
        self.log(
            "epoch_val_loss",
            _sum / ct,
            sync_dist=True
        )
        self.log(
            "epoch_val_f1",
            f1_sum / ct,
            sync_dist=True
        )
        self.log(
            "epoch_val_em",
            em_sum / ct,
            sync_dist=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
