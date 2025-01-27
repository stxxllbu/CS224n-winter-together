#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.
        # print(input.shape) #(length=4, b=5)
        x = self.decoderCharEmb(input)
        # print(x.shape) #(4,5,e=3)
        enc_hidden, (last_dec_hidden, last_dec_cell) = self.charDecoder(x, dec_hidden)
        # print(enc_hidden[-1])#(4,5,4)
        # print(last_dec_hidden)#(1,5,4)
        dec_hidden = (last_dec_hidden, last_dec_cell)
        logits = self.char_output_projection(enc_hidden)
        return logits, dec_hidden
        # logits size=(len, b, vocab)
        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        length=m_word, batch_size = src_len*batch

        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss

        logits, dec_hidden = self.forward(char_sequence[:-1], dec_hidden)
        # logits shape: [len, b, vocab_size]
        # dec_hidden shape: 2 x (1, b, h)

        loss_fcn = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char_pad, reduction='sum')
        score_len, batch = logits.shape[0], logits.shape[1]
        scores = logits.reshape(score_len * batch, -1)

        target_scores = char_sequence[1:].reshape(-1)
        output = loss_fcn(scores, target_scores)

        # Compute log probability of generating true target words
        # P = F.log_softmax(logits, dim=-1)
        # log_prob = torch.gather(P, index=char_sequence[1:].unsqueeze(-1), dim=-1).squeeze(
        #     -1)
        # scores = log_prob.sum()  #
        return output

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        dec_hidden = initialStates  # 2* (1, b, h)
        batch = dec_hidden[0].shape[1]
        output_word = torch.tensor([]*batch, dtype=torch.long, device=device).contiguous()

        # current_char size (1, b)
        current_char = torch.tensor([self.target_vocab.start_of_word]*batch,
                                    dtype=torch.long,device=device).reshape(1, -1).contiguous()

        for t in range(max_length):
            logits, dec_hidden = self.forward(current_char, dec_hidden)
            pt = F.softmax(logits, dim=2)
            current_char = torch.argmax(pt, dim=2)
            output_word = torch.cat((output_word, current_char), dim=0)# (len, b)

        output_word = output_word.permute(1, 0).tolist() #List[List(int)], (b, len) #len指的是word len里面char总数
        ## truncate???
        # if current_char == torch.tensor([self.target_vocab.end_of_word]):
        #     break
        truncated_word = []
        for w in output_word:
            if self.target_vocab.end_of_word in w:
                w = w[:w.index(self.target_vocab.end_of_word)]
            truncated_word.append(w)
        decode_words = [[self.target_vocab.id2char[ch] for ch in w] for w in truncated_word]
        decode_words = ["".join(ch) for ch in decode_words]
        return decode_words



        ### END YOUR CODE

