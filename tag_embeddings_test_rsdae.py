
'''


    Read more about Recursive Neural Language Architecture for Tag Prediction : http://arxiv.org/abs/1603.07646
        

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cbow_tag_embeddings.py


'''
from __future__ import absolute_import
from __future__ import print_function

import collections
import numpy as np
import random
import theano
import sys, string
from six.moves import cPickle
import os, re, json
from sklearn.metrics import label_ranking_average_precision_score

from keras.preprocessing import sequence, text
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils, generic_utils, test_utils
from keras.models import Sequential
from keras.layers.embeddings import WordMultiContextProduct, Embedding, WordTagContextProduct, WordTagContextProduct_tensor
from six.moves import range
from six.moves import zip

max_features = 50000  # vocabulary size: top 50,000 most common words in data
skip_top = 100  # ignore top 100 most common words
nb_epoch = 30
dim_proj = 256  # embedding space dimension
test_labels = 0
train_negative_samples = 4 
test_at_epoch = 5
keep_ex = 10000


save = True
load_model = False
load_tokenizer = False
train_model = True
save_dir = "./models/"
#if not os.path.exists(save_dir):
#    os.makedirs(save_dir)
save_suffix = "aff.10"
model_load_fname = "words_cbow_model.pkl"
model_save_fname = "tags_"+save_suffix+".pkl"
w_tokenizer_fname = "words_tokenizer.pkl"
t_tokenizer_fname = "tags_tokenizer.pkl"

data_path = "./data/docs"

# text preprocessing utils
html_tags = re.compile(r'<.*?>')
to_replace = [('&#x27;', "'")]
hex_tags = re.compile(r'&.*?;')

def get_min_divisor(n):
    for i in range(4,20):
        if n%i == 0:
            return i
    for i in range(2,5):
        if n%i == 0:
            return i

def tag_filter():
    f = string.punctuation
    f = f.replace("'", '')
    f = f.replace("_", '')
    f += '\t\n'
    return f

def clean_comment(comment):
    c = str(comment.encode("utf-8"))
    c = html_tags.sub(' ', c)
    for tag, char in to_replace:
        c = c.replace(tag, char)
    c = hex_tags.sub(' ', c)
    return c


def text_generator_json(path=data_path):
    f = open(path)
    for i, l in enumerate(f):
        comment_data = json.loads(l)
        comment_text = comment_data["comment_text"]
        comment_text = clean_comment(comment_text)
        if i % 10000 == 0:
            print(i)
        yield comment_text
    f.close()
    
def text_generator_tag_format(path=data_path):
    f = open(path)
    for i, l in enumerate(f):
        comment_text = ""
        if len(l.split("<=>"))>1:
            comment_text = l.split("<=>")[-1]
        if i == keep_ex: ## For Quick Testing
            break
        yield comment_text
    f.close()
    
def tag_generator_tag_format(path=data_path):
    f = open(path)
    for i, l in enumerate(f):
        comment_tags = ""
        if len(l.split("<=>"))>1:
            comment_tags = l.split("<=>")[1].split(",")
        if i == keep_ex: ## For Testing
            break
        #if i % 10000 == 0:
        #    print(i)
        yield " ".join(comment_tags)
    f.close()
    

def tag_splits(tag_tokenizer, tag_generator_tag_format):
    train_tags=[]
    test_tags=[]
    
    for i, tags in enumerate(tag_tokenizer.texts_to_sequences_generator(tag_generator_tag_format())):
        trains=[]
        tests=[]
        if len(tags)>3:
            for t in tags:
                if random.random()>0.33:
                    trains.append(t)
                else:
                    tests.append(t)
        else:
            trains = tags
            tests = []
        train_tags.append(trains)
        test_tags.append(tests)
    return train_tags, test_tags

def doc_splits(tag_tokenizer, tag_generator_tag_format):
    train_tags=[]
    test_tags=[]
    train_dict = collections.defaultdict(lambda: collections.defaultdict(int))
    test_dict = collections.defaultdict(lambda: collections.defaultdict(int))
    P_global = 10
    tag_docs = collections.defaultdict(lambda: collections.defaultdict(int))
    for i, tags in enumerate(tag_tokenizer.texts_to_sequences_generator(tag_generator_tag_format())):
        for t in tags:
            tag_docs[t][i]+=1
    max_doc_id = i
    #np.random.seed(0)
    for t in tag_docs.keys():
        tag_doc_arr = tag_docs[t].keys()
        np.random.shuffle(tag_doc_arr)
        for i,d in enumerate(tag_doc_arr):
            if i < P_global:
                train_dict[d][t]+=1
            else:
                test_dict[d][t]+=1
        
    for d in xrange(max_doc_id+1):
        if d in train_dict:
            train_tags.append(train_dict[d].keys())
        else:
            train_tags.append([])
            
        if d in test_dict:
            test_tags.append(test_dict[d].keys())
        else:
            test_tags.append([])    
            
    return train_tags, test_tags

# model management
if load_tokenizer:
    print('Load tokenizer...')
    tokenizer = cPickle.load(open(os.path.join(save_dir, w_tokenizer_fname), 'rb'))
    tag_tokenizer = cPickle.load(open(os.path.join(save_dir, t_tokenizer_fname), 'rb'))
else:
    print("Fit tokenizer...")
    tokenizer = text.Tokenizer(nb_words=max_features)
    tokenizer.fit_on_texts(text_generator_tag_format())
    #print(len(tokenizer.word_index))
    ## fit tag tokenizer
    tag_tokenizer = text.Tokenizer(nb_words=max_features, filters = tag_filter())
    tag_tokenizer.fit_on_texts(tag_generator_tag_format(), prune = 2)
    
    print('len(tag_tokenizer.word_index):',len(tag_tokenizer.word_index))
    #sys.exit(0)
    #print(tag_tokenizer.word_index)
    #sys.exit(0)
    if save:
        print("Save tokenizer...")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cPickle.dump(tokenizer, open(os.path.join(save_dir, w_tokenizer_fname), "wb"))
        cPickle.dump(tag_tokenizer, open(os.path.join(save_dir, t_tokenizer_fname), "wb"))

#if load_model:
#    print('Load model...')
#    model = cPickle.load(open(os.path.join(save_dir, model_load_fname), 'rb'))
# training process
if train_model:
    if load_model:
        print('Load model...')
        model = cPickle.load(open(os.path.join(save_dir, model_load_fname), 'rb'))
    else:
        print('Build model...')
        tag_layer = WordTagContextProduct(max_features, proj_dim=dim_proj, init="normal", neg_samples = 4, weights = None)
        tag_layer.init_tags(tokenizer.word_index, tag_tokenizer.word_index)
        model = Sequential()
        model.add(tag_layer)
        #rmsprop = RMSprop(lr=0.001)
        #sgd = SGD(lr = 0.025, decay = 0.05)
        model.compile(loss='mse', optimizer='adam') ## Adam converges fastest among SGD and RMSprop

    sampling_table = sequence.make_sampling_table(max_features)
    train_tags, test_tags = doc_splits(tag_tokenizer, tag_generator_tag_format)
    fout = open("result."+save_suffix, "w")
    results_agg_mrr, results_agg_lrap, results_agg_prec, results_agg_rec = "","","",""
    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print(model.optimizer.get_config())    
        test_tag_iters = iter(test_tags)
        train_tag_iters = iter(train_tags)
        
        progbar = generic_utils.Progbar(tokenizer.document_count)
        if e > 0 and e % test_at_epoch == 0:
            ##test
            print("\nTesting at iteration", e)
            test_docs, samples_seen = 0, 0
            lraps, losses, test_losses, mrrs, recalls, precs = [], [], [], [], [], [],
            
            for i, _seq in enumerate(tokenizer.texts_to_sequences_generator(text_generator_tag_format())):
                test_tag_seq = next(test_tag_iters)
                train_tag_seq = next(train_tag_iters)
                if len(test_tag_seq)==0:
                    continue
                seq = [s for s in _seq if sampling_table[s] > random.random()]
                if len(seq)==0:
                    continue
                test_docs+=1
                couples, labels = sequence.tagged_cbows(seq, test_tag_seq, len(tag_tokenizer.word_index)+1, 
                                                        window_size=4, negative_samples=test_labels, sampling_table=None)
                tag_couples, labels = sequence.tagged_cbows(seq, test_tag_seq, len(tag_tokenizer.word_index)+1, # replace seq with train_tag_seq
                                                        window_size=4, negative_samples=test_labels, sampling_table=None)
                
                couples = np.array(couples, dtype="int32")
                tag_couples = np.array(tag_couples, dtype="int32")
                labels = np.array(labels)
                X_w = np.array(np.split(couples, len(seq)))
                X_t = np.array(np.split(tag_couples, len(seq)))
                if test_labels ==0 :
                    # Divide number of examples to rank so that GPU does not cause out of memory error
                    splitter = get_min_divisor(len(labels))
                    test_y = np.reshape(np.empty_like(labels, dtype = 'float32'),(labels.shape[0],1))
                    for j in range(splitter):
                        test_loss, test_y_block = model.test_on_batch([X_w[:,j*(labels.shape[0]/splitter): (j+1)*(labels.shape[0]/splitter) ,:], 
                                                                       X_t[:,j*(labels.shape[0]/splitter): (j+1)*(labels.shape[0]/splitter) ,:]],
                                                                labels[j*(labels.shape[0]/splitter): (j+1)*(labels.shape[0]/splitter)]) 
                        test_y[j*(labels.shape[0]/splitter): (j+1)*(labels.shape[0]/splitter)] = test_y_block
                else:
                    test_loss, test_y = model.test_on_batch([X_w, X_t], labels) 
                
                lraps.append(label_ranking_average_precision_score(np.reshape(np.array(labels),test_y.shape).T , test_y.T))
                mrr, recall, prec = test_utils.print_accuracy_results(np.array(labels) , np.reshape(test_y, np.array(labels).shape))
                mrrs.append(mrr)
                recalls.append(recall)
                precs.append(prec)
                losses.append(test_loss)
                test_losses.append(test_loss)
                if len(losses) % 100 == 0:
                    progbar.update(i, values=[("loss", np.sum(losses))])
                    losses = []
                samples_seen += len(labels)
        
            print("\nSkipped="+str(skipped))        
            print("\nlrap="+"{0:.5f}".format(np.mean(np.array(lraps)))+" :loss=" + str(np.mean(test_losses)) + " :Samples seen="+str(test_docs)+ "\n")
            print("mrr=" + "{0:.5f}".format(np.mean(mrrs))+"\n")
            
            str_recall = ",".join(["{0:.5f}".format(r) for r in np.mean(np.array(recalls), axis = 0)])
            print("recall=" + str_recall + "\n")
            
            str_prec = ",".join(["{0:.5f}".format(r) for r in np.mean(np.array(precs), axis = 0)])
            print("precision=" + str_prec + "\n")
            fout.write("------------------------------------------------\n")
            fout.write("iter="+str(e)+":lrap=" + "{0:.5f}".format(np.mean(np.array(lraps)))
                       +":mrr=" + "{0:.5f}".format(np.mean(mrrs)) +"\nrecalls=" + str_recall
                       +"\nprecisions=" + str_prec+ "\n")
            fout.flush()
        # Train
        samples_seen = 0
        losses, lraps = [], []
        tag_iters = iter(train_tags)#tag_tokenizer.texts_to_sequences_generator(tag_generator_tag_format())
        for i, _seq in enumerate(tokenizer.texts_to_sequences_generator(text_generator_tag_format())):
            tag_seq = next(tag_iters)
            seq = [s for s in _seq if sampling_table[s] > random.random()]

            # get skipgram couples for one text in the dataset
            neg_exs = random.sample(range(1, len(tag_tokenizer.word_index)),int(train_negative_samples))
            couples, labels = sequence.tagged_cbows(seq, tag_seq, len(tag_tokenizer.word_index)+1, 
                                                    window_size=4, negative_samples=train_negative_samples, per_tag = True, sampling_table=None,neg_tags = neg_exs )
            tag_couples, labels = sequence.tagged_cbows(tag_seq, tag_seq, len(tag_tokenizer.word_index)+1, 
                                                    window_size=4, negative_samples=train_negative_samples, per_tag = True, sampling_table=None, neg_tags = neg_exs)
            if couples:
                X_w = np.array(np.split(np.array(couples, dtype="int32"), len(seq)))
                X_t = np.array(np.split(np.array(tag_couples, dtype="int32"), len(tag_seq)))
                loss = model.train_on_batch([X_w,X_t], labels)
                losses.append(loss)
                if len(losses) % 100 == 0:
                    progbar.update(i, values=[("loss", np.sum(losses))])
                    losses = []
                samples_seen += len(labels)
                                 
         
        
        
        
        print('\nSamples seen:', samples_seen)
    print("Training completed!")
    fout.close()
    if save:
        print("Saving model...")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cPickle.dump(model, open(os.path.join(save_dir, model_save_fname), "wb"))


print("It's test time!")

# recover the embedding weights trained with skipgram:
weights = model.layers[0].get_weights()[0]

# we no longer need this
del model

weights[:skip_top] = np.zeros((skip_top, dim_proj))
norm_weights = np_utils.normalize(weights)

word_index = tokenizer.word_index
reverse_word_index = dict([(v, k) for k, v in list(word_index.items())])


def embed_word(w):
    i = word_index.get(w)
    if (not i) or (i < skip_top) or (i >= max_features):
        return None
    return norm_weights[i]


def closest_to_point(point, nb_closest=10):
    proximities = np.dot(norm_weights, point)
    tups = list(zip(list(range(len(proximities))), proximities))
    tups.sort(key=lambda x: x[1], reverse=True)
    return [(reverse_word_index.get(t[0]), t[1]) for t in tups[:nb_closest]]


def closest_to_word(w, nb_closest=10):
    i = word_index.get(w)
    if (not i) or (i < skip_top) or (i >= max_features):
        return []
    return closest_to_point(norm_weights[i].T, nb_closest)


''' the resuls in comments below were for:
    5.8M HN comments
    dim_proj = 256
    nb_epoch = 2
    optimizer = rmsprop
    loss = mse
    max_features = 50000
    skip_top = 100
    negative_samples = 1.
    window_size = 4
    and frequency subsampling of factor 10e-5.
'''

'''words = [
    "article",  # post, story, hn, read, comments
    "3",  # 6, 4, 5, 2
    "two",  # three, few, several, each
    "great",  # love, nice, working, looking
    "data",  # information, memory, database
    "money",  # company, pay, customers, spend
    "years",  # ago, year, months, hours, week, days
    "android",  # ios, release, os, mobile, beta
    "javascript",  # js, css, compiler, library, jquery, ruby
    "look",  # looks, looking
    "business",  # industry, professional, customers
    "company",  # companies, startup, founders, startups
    "after",  # before, once, until
    "own",  # personal, our, having
    "us",  # united, country, american, tech, diversity, usa, china, sv
    "using",  # javascript, js, tools (lol)
    "here",  # hn, post, comments
]'''
words = ["latent", "learning", "mining","graphical","privacy","mri","model", "hidden", "neural", "text", "image", "experiment", "data", "mine"]

for w in words:
    res = closest_to_word(w)
    print('====', w)
    for r in res:
        print(r)
