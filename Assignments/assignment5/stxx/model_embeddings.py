#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab, e_char=30, padding=1, kernel_size=5, dropout_rate=0.3):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()
        self.vocab = vocab
        self.word_embed_size = word_embed_size
        self.e_char = e_char
        self.padding = padding
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.char_pad_token_idx = self.vocab.char_pad

        self.charEmbedding = nn.Embedding(len(self.vocab.char2id), self.e_char, padding_idx=self.char_pad_token_idx)
        self.CNN = CNN(
            e_char=self.e_char,
            filters=self.word_embed_size,
            padding=self.padding,
            kernel_size=self.kernel_size
        )
        self.Highway = Highway(
            eword_size=self.word_embed_size
        )
        self.dropout = nn.Dropout(self.dropout_rate)

        ### YOUR CODE HERE for part 1h

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        # input size: (sentence_len, batch, m_word)
        x_emb = self.charEmbedding(input) # x_emb size (sentence_len, batch, m_word, e_char)
        sentence_len, batch, m_word, e_char = x_emb.shape
        x_emb_post = x_emb.reshape(sentence_len*batch, m_word, e_char)
        # print('x_emb_post', x_emb_post.shape)

        xconv_out = self.CNN(x_emb_post)
        # print ('CNN complete', xconv_out.shape)

        xhighway = self.Highway(xconv_out)
        x_dropout = self.dropout(xhighway)
        x_wordemb = x_dropout.reshape(sentence_len, batch, -1)


        return x_wordemb # x_wordemb size = (sent_len, batch, word_embed_size/filter)
        ### END YOUR CODE


