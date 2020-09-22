import pkg_resources
idf_path = pkg_resources.resource_filename(__name__, 'assets/idf-vals.parquet')

import spacy
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import os
nlp = spacy.load("en")

package_directory = os.path.dirname(os.path.abspath(__file__))

class AUT_Scorer:
    
    def __init__(self, model_dict=None):
        self._idf_ref = None
        self._models = dict()
        
        if model_dict:
            self._preload_models[model_dict]
        
    def load_model(self, name, path, format='default', custom_parser=False, mmap='r'):
        ''' Load a model into memory.
        Models should in Gensim's wordvectors format. You can save to this format
        from any other format loaded in Gensim with 'save'.
        '''
        if format == 'default':
            self._models[name] = KeyedVectors.load(path, mmap=mmap)
        elif format == 'word2vec':
            self._models[name] = KeyedVectors.load_word2vec_format(path, binary=True)
    
    @property
    def models(self):
        ''' Return just the names of the models'''
        return list(self._models.keys())
        
    def _preload_models(self, model_dict):
        '''
        Preload models from a list of dicts, where the each dict item has the arguments
        for _load_model: e.g. [model-path'}
        '''
        for model in model_dict:
            # Expand dict argument
            self.load_model(**model)
    
    def fluency(self, **kwargs):
        raise Exception("Fluency is not calculated at the item level. Use `ocs.file.fluency` to calculate it.")
    
    def elaboration(self, phrase, elabfunc="whitespace"):
        if elabfunc == 'whitespace':
            elabfunc = lambda x: len(x.split())
        elif elabfunc == 'tokenized':
            elabfunc = lambda x: len([word for word in nlp(x[:nlp.max_length], disable=['tagger', 'parser', 'ner']) if not word.is_punct])
        elif elabfunc == 'idf':
            def idf_elab(phrase):
                phrase = nlp(phrase[:nlp.max_length], disable=['tagger', 'parser', 'ner'])
                weights = []
                for word in phrase:
                    if word.is_punct:
                        continue
                    weights.append(self.idf[word.lower_] if word.lower_ in self.idf else self.default_idf)
                return sum(weights)
            elabfunc = idf_elab
        elif elabfunc == "stoplist":
            def stoplist_elab(phrase):
                phrase = nlp(phrase[:nlp.max_length], disable=['tagger', 'parser', 'ner'])
                non_stopped = [word for word in phrase if not (word.is_stop or word.is_punct)]
                return len(non_stopped)
            elabfunc = stoplist_elab
        elif elabfunc == "pos":
            def pos_elab(phrase):
                phrase = nlp(phrase[:nlp.max_length], disable=['parser', 'ner'])
                remaining_words = [word for word in phrase if (word.pos_ in ['NOUN','VERB','ADJ', 'ADV', 'PROPN']) and not word.is_punct]
                return len(remaining_words)
            elabfunc = pos_elab

        try:
            elab = elabfunc(phrase)
        except:
            raise
            elab = None
        return elab
    
    @property
    def idf(self):
        ''' Load IDF scores. Uses the page level scores from 
        
        Organisciak, P. 2016. Term Frequencies for 235k Language and Literature Texts. 
            http://hdl.handle.net/2142/89515.
        '''
        if not self._idf_ref:
            idf_df = pd.read_parquet(idf_path)
            self._idf_ref = idf_df['IPF'].to_dict()
            # for the default NA score, use something around 10k.
            self.default_idf = idf_df.iloc[10000]['IPF']
        return self._idf_ref
    
    def _get_phrase_vecs(self, phrase, model, stopword=False, term_weighting=False):
        ''' Return a stacked array of model vectors. Phrase can be a Spacy doc'''
        
        arrlist = []
        weights = []
        
        # Response should be a spacy doc
        if type(phrase) != spacy.tokens.doc.Doc:
            phrase = nlp(phrase[:nlp.max_length], disable=['tagger', 'parser', 'ner'])

        for word in phrase:
            if stopword and word.is_stop:
                continue
            else:
                try:
                    vec = self._models[model][word.lower_]
                    arrlist.append(vec)
                except:
                    continue

                if term_weighting:
                    weight = self.idf[word.lower_] if word.lower_ in self.idf else self.default_idf
                    weights.append(weight)
        
        if len(arrlist):
            vecs = np.vstack(arrlist)
            return vecs, weights
        else:
            return [], []
    
    
    def originality(self, target, response, model='first',
                    stopword=False, term_weighting=False, flip=True):
        '''
        Score originality.
        '''
        scores = []
        weights = []

        if model not in self._models:
            if (len(self._models) == 1) or (model == 'first'):
                # Use only loaded mode;
                model = list(self._models.keys())[0]
            else:
                raise Exception('No model loaded by that name')
        
        vecs, weights = self._get_phrase_vecs(response, model, stopword, term_weighting)
        
        if len(vecs) is 0:
            return None
        
        if ' ' in target:
            targetvec = self._get_phrase_vecs(target, model, stopword, term_weighting)[0].sum(0)
        else:
            targetvec = self._models[model][target.lower()]
            
        scores = self._models[model].cosine_similarities(targetvec, vecs)
        
        if len(scores) and not term_weighting:
            s = np.mean(scores)
        elif len(scores):
            s = np.average(scores, weights=weights)
        else:
            return None
        
        if flip:
            s = 1 - s
        return s
        
        
def combine_elmo_layers(vectors):
    '''
    Combine the 3 ELMO layers into 1 wide layer.
    
    Output: a 3072 x n_words vector.
    '''
    a = np.rollaxis(vectors, 2, 1)
    b = np.reshape(a, (3*1024, a.shape[2]))
    c = np.rollaxis(b, 1)
    return c

def originality_elmo(target, response):
    try:
        promptvec = combine_layers(elmo.embed_sentence(target.split())).sum(0)

        wordvecs = elmo.embed_sentence(response.split())
        wordvecs = combine_elmo_layers(wordvecs)
        resvec = wordvecs.sum(0)

        dist = scipy.spatial.distance.cosine(promptvec, resvec)
        return dist
    except:
        return None

class elmo_model():
    
    '''
    A wrapper to align using ELMO with the API of Gensim wordsim
    '''
    
    def __init__(self):
        from allennlp.commands.elmo import ElmoEmbedder
        self.elmo = ElmoEmbedder()
        
    def similarity(self, target, response):
        try:
            promptvec = combine_layers(elmo.embed_sentence(target.split())).sum(0)

            wordvecs = elmo.embed_sentence(response.split())
            wordvecs = combine_elmo_layers(wordvecs)
            resvec = wordvecs.sum(0)

            dist = scipy.spatial.distance.cosine(promptvec, resvec)
            return dist
        except:
            return None

def originality_row(x, **kwargs):
    response = nlp(str(x['response'])[:nlp.max_length], disable=['tagger', 'parser', 'ner'])
    return originality(x['prompt'].lower(), response, **kwargs)