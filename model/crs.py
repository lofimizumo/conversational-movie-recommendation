from torch import optim, nn
from util.dataset import *
import torch
import pytorch_lightning as pl
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertModel, BertTokenizer, BertConfig, BartTokenizer, BartModel
from tqdm import trange
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from functools import reduce


class LitRecBartModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.p_encoder = BartModel.from_pretrained('facebook/bart-base')
        self.q_encoder = self.p_encoder
        self.pretrained_movie_embedding = nn.Embedding(2000, 768)
        self.p_fc = MyTransformer()
        self.q_fc = self.p_fc
        
        for param in self.p_encoder.parameters():
            param.requires_grad = False


        # for param in self.q_encoder.parameters():
        #     param.requires_grad = False

        # for param in self.q_encoder.encoder.layer[-3:].parameters():
        #     param.requires_grad = True

    def p_forward(self, batch):
        """
        movie plot encoder
        Args:
            batch: tokenized movie plots.
        Returns:
            movie plot embeddings
        """
        self.p_encoder = self.p_encoder
        self.p_fc = self.p_fc
        batch = {key: tensor.cuda() for key, tensor in batch.items()}
        output = self.p_encoder(**batch).last_hidden_state
        ret = self.p_fc(output)
        return ret

    def q_forward(self, batch):
        """
        question encoder
        Args:
            batch: tokenized question.
        Returns:
            question embeddings
        """
        self.q_encoder = self.q_encoder
        self.q_fc = self.q_fc
        batch = {key: tensor.cuda() for key, tensor in batch.items()}
        output = self.q_encoder(**batch).last_hidden_state
        ret = self.q_fc(output)
        return ret

    def training_step(self, batch):
        pos_neg_plots, q = batch
        pos_plots = pos_neg_plots[0]
        neg_plots = pos_neg_plots[1]
        pos_plots = self.tokenizer(pos_plots, max_length=512,
                                   padding='max_length', return_tensors='pt')
        neg_plots = self.tokenizer(neg_plots, max_length=512,
                                   padding='max_length', return_tensors='pt')
        q = self.tokenizer(list(q), max_length=512,
                           padding='max_length', return_tensors='pt')
        pos_embs = self.p_forward(pos_plots)
        neg_embs = self.p_forward(neg_plots)
        q_embs = self.q_forward(q)
        # pos_scores = torch.bmm(pos_embs.unsqueeze(
        #     1), q_embs.unsqueeze(2)).squeeze()
        # neg_scores = torch.bmm(neg_embs.unsqueeze(
        #     1), q_embs.unsqueeze(2)).squeeze()
        pos_scores = F.cosine_similarity(pos_embs, q_embs)
        neg_scores = F.cosine_similarity(neg_embs, q_embs)

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

        batch_size = 2
        num_batches = len(plots) // batch_size

        num_batches_testonly = 1

        embs_list = []
        print('------pretraining movie embeddings...-------')
        with torch.no_grad():
            for i in trange(num_batches):
                batch_start = i * batch_size
                batch_end = batch_start + batch_size
                batch = plots[batch_start:batch_end]
                batch = self.tokenizer(batch, max_length=512,
                       padding='max_length', return_tensors='pt')
                
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
        q = self.q_forward(question)
        movie_database_embs = self.pretrained_movie_embedding.weight
        movie_database_embs = movie_database_embs.permute(1, 0)
        print(
            f'movie_database_embs.shape:{movie_database_embs.shape}\nq.shape:{q.shape}')
        distance = torch.mm(q, movie_database_embs)
        idxs = distance.topk(k=10)[1]
        return idxs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



class LitRecBertModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.config = BertConfig()
        self.p_encoder = BertModel(self.config, add_pooling_layer=False)
        self.q_encoder = self.p_encoder
        self.pretrained_movie_embedding = nn.Embedding(2000, 768)
        self.p_fc = MyTransformer()
        self.q_fc = self.p_fc
        # self.q_fc = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 768)
        # )
        for param in self.p_encoder.parameters():
            param.requires_grad = False

        for param in self.p_encoder.encoder.layer[-3:].parameters():
            param.requires_grad = True

        # for param in self.q_encoder.parameters():
        #     param.requires_grad = False

        # for param in self.q_encoder.encoder.layer[-3:].parameters():
        #     param.requires_grad = True

    def p_forward(self, batch):
        """
        movie plot encoder
        Args:
            batch: tokenized movie plots.
        Returns:
            movie plot embeddings
        """
        self.p_encoder = self.p_encoder
        self.p_fc = self.p_fc
        batch = {key: tensor.cuda() for key, tensor in batch.items()}
        bert_output = self.p_encoder(**batch).last_hidden_state
        ret = self.p_fc(bert_output)
        return ret

    def q_forward(self, batch):
        """
        question encoder
        Args:
            batch: tokenized question.
        Returns:
            question embeddings
        """
        self.q_encoder = self.q_encoder
        self.q_fc = self.q_fc
        batch = {key: tensor.cuda() for key, tensor in batch.items()}
        bert_output = self.q_encoder(**batch).last_hidden_state
        ret = self.q_fc(bert_output)
        return ret

    def training_step(self, batch):
        pos_neg_plots, q = batch
        pos_plots = pos_neg_plots[0]
        neg_plots = pos_neg_plots[1]
        pos_plots = self.tokenizer(pos_plots, max_length=512,
                                   padding='max_length', return_tensors='pt')
        neg_plots = self.tokenizer(neg_plots, max_length=512,
                                   padding='max_length', return_tensors='pt')
        q = self.tokenizer(list(q), max_length=512,
                           padding='max_length', return_tensors='pt')
        pos_embs = self.p_forward(pos_plots)
        neg_embs = self.p_forward(neg_plots)
        q_embs = self.q_forward(q)
        # pos_scores = torch.bmm(pos_embs.unsqueeze(
        #     1), q_embs.unsqueeze(2)).squeeze()
        # neg_scores = torch.bmm(neg_embs.unsqueeze(
        #     1), q_embs.unsqueeze(2)).squeeze()
        pos_scores = F.cosine_similarity(pos_embs, q_embs)
        neg_scores = F.cosine_similarity(neg_embs, q_embs)

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
        batch_size = 2
        num_batches = len(plots) // batch_size

        num_batches_testonly = 1

        embs_list = []
        print('------pretraining movie embeddings...-------')
        with torch.no_grad():
            for i in trange(num_batches):
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
        print(
            f'movie_database_embs.shape:{movie_database_embs.shape}\nq.shape:{q.shape}')
        distance = torch.mm(q, movie_database_embs)
        idxs = distance.topk(k=10)[1]
        return idxs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class MyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=12),
            num_layers=4
        )
        self.pooler = nn.Linear(512, 1)

    def forward(self, x):
        # Apply transformer
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        # Apply pooling to obtain output tensor of shape (batch_size, hidden_size)
        x = self.pooler(x).squeeze(2)
        return x
