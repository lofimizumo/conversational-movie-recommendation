import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from model.crs import LitRecBartModel, LitRecBertModel
from util.dataset import MoviePlotDataset, MoviePlotDatasetSmall
from collections import OrderedDict
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from util.dataset import *
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='best_model',
    monitor='train_loss',
    mode='min',
    save_top_k=1
)
early_stop_callback = EarlyStopping(
    monitor='train_loss',
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode='min'
)

torch.set_float32_matmul_precision('medium')
dataset = MoviePlotDataset('plot', 'question')
train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=8)
crs_model = LitRecBartModel()
trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available()
                     else None, max_epochs=1, callbacks=early_stop_callback)
trainer.fit(model=crs_model, train_dataloaders=train_dataloader)
crs_model.cuda()
crs_model.pretrain_movie_embs('data/movie_1000_clean.csv')


movie_dict = read_plots_from_csv('data/movie_1000_clean.csv')
names = movie_dict.keys()
idx = list(range(len(names)))
idx_name_dict = OrderedDict(zip(idx, names))


def pred(question):
    ret = crs_model.inference(question)
    ret = ret.reshape(-1)
    ret = list(ret)
    ret = [int(x) for x in ret]
    pred_names = [idx_name_dict[int(x)] for x in ret]
    print(f'Question:{question}\nMovie:{pred_names}\n')


questions = ['can you recommend a movie about dinosaur',

             'can you recommend a movie about alien fighting human',

             'can you recommend a movie about love story',

             'can you recommend a movie about ghost and spirit',

             'can you recommend a movie about animals',

             'can you recommend a movie about war between countries',

             'can you recommend a movie about kids and families',

             'can you recommend a movie about kids tring to save the alien',

             'can you recommend a movie about animals fighting each other for power',

             'can you recommend a movie about a thrill adventure']


for i in questions:
    pred(i)
