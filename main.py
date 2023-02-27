import pytorch_lightning as pl
from torch.utils.data import DataLoader
from model.crs import LitRecModel
from util.dataset import MoviePlotDataset, MoviePlotDatasetSmall


dataset = MoviePlotDataset('plot', 'question')
train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
crs_model = LitRecModel()
trainer = pl.Trainer(limit_train_batches=20, max_epochs=2)
trainer.fit(model=crs_model, train_dataloaders=train_dataloader)


question = 'can you recommend a movie about human fighting aliens'
ret = crs_model.inference(question)
print(ret)
#%%
