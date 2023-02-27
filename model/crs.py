from torch import optim, nn
from util.dataset import *
import torch
import pytorch_lightning as pl
import pandas as pd
from transformers import BertModel, AutoTokenizer, BertConfig
from tqdm import trange


class LitRecModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.config = BertConfig()
        self.p_encoder = BertModel(self.config)
        self.q_encoder = BertModel(self.config)
        self.pretrained_movie_embedding = nn.Embedding(20000, 768)

    def p_forward(self, batch):
        """
        movie plot encoder
        Args:
            batch: tokenized movie plots.
        Returns:
            movie plot embeddings
        """
        return self.p_encoder(batch).pooler_output

    def q_forward(self, batch):
        """
        question encoder
        Args:
            batch: tokenized question.
        Returns:
            question embeddings
        """
        return self.q_encoder(batch).pooler_output

    def training_step(self, batch):
        pos_neg_plots, q = batch
        pos_plots = torch.cat(pos_neg_plots[:1])
        neg_plots = torch.cat(pos_neg_plots[1:])
        pos_embs = self.p_encoder(pos_plots).pooler_output
        neg_embs = self.p_encoder(neg_plots).pooler_output
        q_embs = self.q_encoder(q).pooler_output
        pos_scores = torch.bmm(pos_embs.unsqueeze(
            1), q_embs.unsqueeze(2)).squeeze()
        neg_scores = torch.bmm(neg_embs.unsqueeze(
            1), q_embs.unsqueeze(2)).squeeze()

        loss = -nn.functional.logsigmoid(pos_scores-neg_scores).sum()
        self.log("train_loss", loss)
        return loss

    def pretrain_movie_embs(self, filename):
        """
        pre-calculate the movie plot embeddings, embeddings are stored for future inference
        Args:
            filename: the csv file storing the movie names and plots
        Returns:
            None
        """
        dict_movie_plots = read_plots_from_csv(filename)
        plots = list(dict_movie_plots.values())
        plots = self.tokenizer(plots, max_length=512,
                               padding='max_length', return_tensors='pt')
        plots = plots['input_ids'].squeeze(0)

        # split movie plots to calculate the embeddings
        batch_size = 5
        num_batches = len(plots) // batch_size

        num_batches_testonly = 1

        embs_list = []
        for i in trange(num_batches_testonly):
            batch_start = i * batch_size
            batch_end = batch_start + batch_size
            batch = plots[batch_start:batch_end]
            embs = self.p_forward(batch)
            embs_list.append(embs)

        embs_weight = torch.cat(embs_list)
        self.pretrained_movie_embedding = self.pretrained_movie_embedding.from_pretrained(
            embs_weight)

    def inference(self, question):
        """
        Return the list of top-k candidate movie recommendation
        Args:
            question: tokenized question describing the desired movie plot
        Returns:
            index of top-k candidate movies
        """
        question = self.tokenizer(question, max_length=512,
                                  padding='max_length', return_tensors='pt')
        question = question['input_ids'].squeeze(0)
        question = question.reshape([1, -1])
        q = self.q_forward(question)
        movie_database_embs = self.pretrained_movie_embedding.weight
        movie_database_embs = movie_database_embs.permute(1, 0)
        distance = torch.mm(q, movie_database_embs)
        movie_idx = distance.argmax()
        return movie_idx

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
