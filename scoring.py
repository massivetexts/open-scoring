import spacy
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
nlp = spacy.load("en")

class AUT_Scorer:
    
    def __init__(self, model_dict=None):
        self._idf_ref = None
        self._models = dict()
        
        if model_dict:
            self._preload_models[model_dict]
        
        
    def load_model(self, name, path, custom_parser=False):
        ''' Load a model into memory. Models should either be converted to word2vec
        binary format (using Gensim's command line tool, then the convertvec code)
        or accompany a custom parser (currently unimplemented since refactor).
        '''
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
    
    @property
    def idf(self):
        ''' Load IDF scores. Uses the page level scores from {ANON}. To use them,
        download that dataset, open the csv, and write the IPF and token columns
        to parquet in the data/idf-vals.parquet. e.g.
        
        import pandas as pd
        pd.read_csv('idf.csv')[['token', 'IPF']].to_parquet('data/idf-vals.parquet')
        '''
        if not self._idf_ref:
            idf_df = pd.read_parquet('data/idf-vals.parquet').set_index('token')
            self._idf_ref = idf_df['IPF'].to_dict()
            # for the default NA score, use something around 10k.
            self.default_idf = idf_df.iloc[10000]['IPF']
        return self._idf_ref
    
    def originality(self, target, response, model,
                    stopword=False, term_weighting=False, flip=True):
        '''
        Score originality.
        '''
        scores = []
        weights = []

        if model not in self._models:
            raise Exception('No model loaded by that name')
        
        # Response should be a spacy doc
        if type(response) != spacy.tokens.doc.Doc:
            response = nlp(response, disable=['tagger', 'parser', 'ner'])
        
        for word in response:
            if stopword and word.is_stop:
                continue

            try:
                sim = self._models[model].similarity(target.lower(),  word.lower_)
                scores.append(sim)
            except:
                continue

            if term_weighting:
                weight = self.idf[word.lower_] if word.lower_ in self.idf else self.default_idf
                weights.append(weight)

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
    response = nlp(str(x['response']), disable=['tagger', 'parser', 'ner'])
    return originality(x['prompt'].lower(), response, **kwargs)