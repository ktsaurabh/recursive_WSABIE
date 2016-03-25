from __future__ import absolute_import
# -*- coding: utf-8 -*-
import numpy as np
import random, sys
from six.moves import range

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """
        Pad each sequence to the same length: 
        the length of the longest sequence.

        If maxlen is provided, any sequence longer
        than maxlen is truncated to maxlen. Truncation happens off either the beginning (default) or
        the end of the sequence.

        Supports post-padding and pre-padding (default).

    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    #print(nb_samples)
    
    if maxlen is None:
        maxlen = np.max(lengths)
    #print(maxlen)
    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def make_sampling_table(size, sampling_factor=1e-5):
    '''
        This generates an array where the ith element
        is the probability that a word of rank i would be sampled,
        according to the sampling distribution used in word2vec.
        
        The word2vec formula is:
            p(word) = min(1, sqrt(word.frequency/sampling_factor) / (word.frequency/sampling_factor))

        We assume that the word frequencies follow Zipf's law (s=1) to derive 
        a numerical approximation of frequency(rank):
           frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))
        where gamma is the Euler-Mascheroni constant.
    '''
    gamma = 0.577
    rank = np.array(list(range(size)))
    rank[0] = 1
    inv_fq = rank * (np.log(rank) + gamma) + 0.5 - 1./(12.*rank)
    f = sampling_factor * inv_fq
    return np.minimum(1., f / np.sqrt(f))


def skipgrams(sequence, vocabulary_size, 
    window_size=4, negative_samples=1., shuffle=True, 
    categorical=False, sampling_table=None):
    ''' 
        Take a sequence (list of indexes of words), 
        returns couples of [word_index, other_word index] and labels (1s or 0s),
        where label = 1 if 'other_word' belongs to the context of 'word',
        and label=0 if 'other_word' is ramdomly sampled

        @param vocabulary_size: int. maximum possible word index + 1
        @param window_size: int. actually half-window. The window of a word wi will be [i-window_size, i+window_size+1]
        @param negative_samples: float >= 0. 0 for no negative (=random) samples. 1 for same number as positive samples. etc.
        @param categorical: bool. if False, labels will be integers (eg. [0, 1, 1 .. ]), 
            if True labels will be categorical eg. [[1,0],[0,1],[0,1] .. ]

        Note: by convention, index 0 in the vocabulary is a non-word and will be skipped.
    '''
    couples = []
    labels = []
    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue

        window_start = max(0, i-window_size)
        window_end = min(len(sequence), i+window_size+1)
        for j in range(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                couples.append([wi, wj])
                if categorical:
                    labels.append([0,1])
                else:
                    labels.append(1)

    if negative_samples > 0:
        nb_negative_samples = int(len(labels) * negative_samples)
        words = [c[0] for c in couples]
        random.shuffle(words)

        couples += [[words[i%len(words)], random.randint(1, vocabulary_size-1)] for i in range(nb_negative_samples)]
        if categorical:
            labels += [[1,0]]*nb_negative_samples
        else:
            labels += [0]*nb_negative_samples

    if shuffle:
        seed = random.randint(0,10e6)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)

    return couples, labels


def cbows(sequence, vocabulary_size, 
    window_size=4, negative_samples=1., shuffle=False, 
    categorical=False, sampling_table=None):
    ''' 
        Take a sequence (list of indexes of words), 
        returns couples of [word_index, other_word index] and labels (1s or 0s),
        where label = 1 if 'other_word' belongs to the context of 'word',
        and label=0 if 'other_word' is ramdomly sampled

        @param vocabulary_size: int. maximum possible word index + 1
        @param window_size: int. actually half-window. The window of a word wi will be [i-window_size, i+window_size+1]
        @param negative_samples: float >= 0. 0 for no negative (=random) samples. 1 for same number as positive samples. etc.
        @param categorical: bool. if False, labels will be integers (eg. [0, 1, 1 .. ]), 
            if True labels will be categorical eg. [[1,0],[0,1],[0,1] .. ]

        Note: by convention, index 0 in the vocabulary is a non-word and will be skipped.
    '''
    couples = []
    labels = []
    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue

        window_start = max(0, i-window_size)
        window_end = min(len(sequence), i+window_size+1)
        couple_w = [wi]
        for j in range(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                couple_w += [wj]
        if categorical:
            labels.append([0,1])
        else:
            labels.append(1)       
        couples.append(couple_w)
    if negative_samples > 0:
        #nb_negative_samples = int(len(labels) * negative_samples)
        words = [c[1:] for c in couples]
        
        #random.shuffle(words)
        for s in words:
            for i in range(int(negative_samples)):
                couple = [random.randint(1, vocabulary_size-1)] + s
                couples.append(couple)
                if categorical:
                    labels += [[1,0]]
                else:
                    labels += [0]
    assert(len(couples) == len(labels))
    if shuffle:
        seed = random.randint(0,10e6)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)

    return couples, labels

def tagged_cbows(sequence, tags, vocabulary_size, 
    window_size=4, negative_samples=1., shuffle=False, per_tag = False,
    categorical=False, sampling_table=None, neg_tags = None):
    ''' 
        Take a sequence (list of indexes of words), 
        returns couples of [word_index, other_word index] and labels (1s or 0s),
        where label = 1 if 'other_word' belongs to the context of 'word',
        and label=0 if 'other_word' is ramdomly sampled

        @param vocabulary_size: int. maximum possible word index + 1
        @param window_size: int. actually half-window. The window of a word wi will be [i-window_size, i+window_size+1]
        @param negative_samples: float >= 0. 0 for no negative (=random) samples. 1 for same number as positive samples. etc.
        @param categorical: bool. if False, labels will be integers (eg. [0, 1, 1 .. ]), 
            if True labels will be categorical eg. [[1,0],[0,1],[0,1] .. ]

        Note: by convention, index 0 in the vocabulary is a non-word and will be skipped.
        Note: If neg sample = -1. take all the examples
    '''
    couples = []
    labels = []
    neg_exs = []
    #print "vocab:",vocabulary_size
    if int(negative_samples) <=0:
        for i, wi in enumerate(sequence):
            if not wi:
                    print "wi is 0!"
            if sampling_table is not None:
                    if sampling_table[wi] < random.random():
                        #padding will take care of rest
                        #tag_couple.append([0,0])                   
                        continue
            for tag in range(0, vocabulary_size):
                couples.append([tag, wi])
        
        labels = [0]*(vocabulary_size)
        #print tags
        for tag in tags:
            if tag >= len(labels):
                print tag,":",len(labels)
            labels[tag] = 1
        return couples, labels
                
    
    if not per_tag:
        ## make tags set
        if int(negative_samples)>0:
            for tag in tags:
                neg_exs.append(tag)
                labels.append(1)
            for _ in range(int(negative_samples)):
                r_tag = random.randint(1, vocabulary_size-1)
                while True:
                    if r_tag in set(tags):
                        r_tag = random.randint(1, vocabulary_size-1)
                    else:
                        break
                    
                neg_exs.append(random.randint(1, vocabulary_size-1))
                labels.append(0)
        for i, wi in enumerate(sequence):
            if not wi:
                    print "wi is 0!"
            if sampling_table is not None:
                    if sampling_table[wi] < random.random():
                        #padding will take care of rest
                        #tag_couple.append([0,0])                   
                        continue
            for tag in neg_exs:
                couples.append([tag, wi])
                
        return couples, labels
    
    if not neg_tags:
        if int(negative_samples)>0:
            for _ in range(int(negative_samples)):
                neg_exs.append(random.randint(1, vocabulary_size-1))
    else:
        neg_exs = neg_tags
    
    for i, wi in enumerate(sequence):
        if not wi:
                print "wi is 0!"
        if sampling_table is not None:
                if sampling_table[wi] < random.random():
                    #padding will take care of rest
                    #tag_couple.append([0,0])                   
                    continue
        for tag in tags:
            couples.append([tag, wi])
            if negative_samples > 0:
                for j in range(len(neg_exs)):
                    couples.append([neg_exs[j], wi])
    
    
    for tag in tags:
        labels.append(1)
        if negative_samples > 0:
            for j in range(len(neg_exs)):
                labels.append(0)
            
    
    
    
    
    
    '''
    for tag in tags:
        tag_couple = []
        for i, wi in enumerate(sequence):
            if not wi:
                print "wi is 0!"
            #if not wi:
            #    continue
            if sampling_table is not None:
                if sampling_table[wi] < random.random():
                    #padding will take care of rest
                    #tag_couple.append([0,0])                   
                    continue
            couple_w = [tag, wi]
            window_start = max(0, i-window_size)
            window_end = min(len(sequence), i+window_size+1)
            
            for j in range(window_start, window_end):
                if j != i:
                    wj = sequence[j]
                    if not wj:
                        continue
                    couple_w += [wj]
            tag_couple.append(couple_w)
        
        if categorical:
            labels.append([0,1])
        else:
            labels.append(1)       
            
        if negative_samples > 0:
            #nb_negative_samples = int(len(labels) * negative_samples)
            words = [c[1:] for c in tag_couple]
            
            #random.shuffle(words)
            for i in range(int(negative_samples)):
                
                r_tag = random.randint(1, vocabulary_size-1)
                for s in words:
                    #if s[0] == 0:
                    #    tag_couple.append([0,0])
                    #    continue
                    couple = [r_tag] + s
                    tag_couple.append(couple)
                
                if categorical:
                    labels += [[1,0]]
                else:
                    labels += [0]
        couples += tag_couple
    
    print "len(couples):",len(couples)
    print "len(sequence)*(int(negative_samples)+1):",len(sequence)*(int(negative_samples)+1)
    print "len(labels):",len(labels)
    assert(len(couples) == len(tags)*len(sequence)*(int(negative_samples)+1))
    assert(len(labels) == len(tags)*(int(negative_samples)+1))'''
        

    return couples, labels



