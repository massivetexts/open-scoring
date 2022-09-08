import pkg_resources
idf_path = pkg_resources.resource_filename(__name__, 'assets/idf-vals.parquet')

import openai
from tqdm import tqdm
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
import os
import getpass
import inflect
import spacy

package_directory = os.path.dirname(os.path.abspath(__file__))

class AUT_Scorer:
    
    def __init__(self, model_dict=None):

        self._idf_ref = None
        self._models = dict()

        self.nlp = spacy.load("en_core_web_sm")
        # for pluralizing
        self.p = inflect.engine()
        
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
            elabfunc = lambda x: len([word for word in self.nlp(x[:self.nlp.max_length], disable=['tagger', 'parser', 'ner', 'lemmatizer']) if not word.is_punct])
        elif elabfunc == 'idf':
            def idf_elab(phrase):
                phrase = self.nlp(phrase[:self.nlp.max_length], disable=['tagger', 'parser', 'ner', 'lemmatizer'])
                weights = []
                for word in phrase:
                    if word.is_punct:
                        continue
                    weights.append(self.idf[word.lower_] if word.lower_ in self.idf else self.default_idf)
                return sum(weights)
            elabfunc = idf_elab
        elif elabfunc == "stoplist":
            def stoplist_elab(phrase):
                phrase = self.nlp(phrase[:self.nlp.max_length], disable=['tagger', 'parser', 'ner', 'lemmatizer'])
                non_stopped = [word for word in phrase if not (word.is_stop or word.is_punct)]
                return len(non_stopped)
            elabfunc = stoplist_elab
        elif elabfunc == "pos":
            def pos_elab(phrase):
                phrase = self.nlp(phrase[:self.nlp.max_length], disable=['parser', 'ner', 'lemmatizer'])
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
    
    def _get_phrase_vecs(self, phrase, model, stopword=False, term_weighting=False, exclude=[]):
        ''' Return a stacked array of model vectors. Phrase can be a Spacy doc
        
        exclude adds additional words to ignore
        '''
        
        arrlist = []
        weights = []
        
        # Response should be a spacy doc
        if type(phrase) != spacy.tokens.doc.Doc:
            phrase = self.nlp(phrase[:self.nlp.max_length], disable=['parser', 'ner', 'lemmatizer'])

        exclude = [x.lower() for x in exclude]
        for word in phrase:
            if stopword and word.is_stop:
                continue
            elif word.lower_ in exclude:
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
                    stopword=False, term_weighting=False, flip=True,
                    exclude_target=False):
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
        
        exclude_words = []
        if exclude_target:
            # assumes that the target prompts are cleanly whitespace-tokenizable (i.e. no periods, etc)
            exclude_words = target.split()
            for word in exclude_words:
                try:
                    sense = self.p.plural(word.lower())
                    if (type(sense) is str) and len(sense) and (sense not in exclude_words):
                        exclude_words.append(sense)
                except:
                    print("Error pluralizing", word)
        vecs, weights = self._get_phrase_vecs(response, model, stopword, term_weighting,
                                              exclude=exclude_words)
        
        if len(vecs) == 0:
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


GPTMODELS = dict(
    gpt="ada:ft-massive-texts-lab:gt-main2-2022-08-01-19-24-54",
    babbage="babbage:ft-massive-texts-lab:gt-main2-2022-08-01-19-26-25",
    curie="curie:ft-massive-texts-lab:gt-main2-2022-08-01-19-44-29",
    davinci="davinci:ft-massive-texts-lab:gt-main2-2022-08-05-16-46-47"
)
class GPT_Scorer:
    def __init__(self, openai_key_path=False, model_dict=False):
        if openai_key_path:
            openai.api_key_path = openai_key_path
        else:
            openai.api_key = getpass.getpass(prompt='Enter API Key:').strip()
        if model_dict:
            self._models = model_dict
        else:
            self._models = GPTMODELS

    def originality(self, target, response, model='first', raise_errs=False, **kwargs):
        if model == 'first':
            model = self.models[0]
        gptprompt = self._craft_gptprompt(target, response)
        score_raw = self._score_gpt(gptprompt, model=model, just_final=True)[0]
        try:
            score = int(score_raw) / 10
        except:
            if raise_errs:
                print(f"GPT prompt: {gptprompt.strip()}")
                print(f"raw response: {score_raw}")
                raise
            score = None
        return score

    def add_model(name, finetunepath):
        self.models['name'] = finetunepath

    def originality_batch(self, targets, responses, model='first', raise_errs=False, batch_size=750, **kwargs):
        scores = []
        assert len(targets) == len(responses)
        if model == 'first':
            model = self.models[0]

        nbatches = np.ceil(len(targets) / batch_size).astype(int)
        for i in tqdm(range(nbatches)):
            targetbatch = targets[i*batch_size:(i+1)*batch_size]
            responsebatch = responses[i*batch_size:(i+1)*batch_size]

            gptprompts = [self._craft_gptprompt(target, response) for target, response in zip(targetbatch, responsebatch)]
            scores_raw = self._score_gpt(gptprompts, model=model, just_final=True)
            for i, score_raw in enumerate(scores_raw):
                try:
                    score = int(score_raw.strip()) / 10
                except:
                    if raise_errs:
                        print(f"GPT prompt: {gptprompts[i].strip()}")
                        print(f"raw response: {score_raw}")
                        raise
                    score = None
                scores.append(score)
        return scores


    @property
    def models(self):
        ''' Return just the names of the models'''
        return list(self._models.keys())

    def fluency(self, **kwargs):
        raise Exception("Fluency is not calculated at the item level. Use `ocs.file.fluency` to calculate it.")
    
    def elaboration(self, phrase, elabfunc="whitespace"):
        raise Exception("Fluency is not calculated for LLM scorer. Use the base AUT scorer.")

    def _craft_gptprompt(self, item, response, prompt_template='aut'):
        # prompt templates should take 2 args - item and response
        if prompt_template == 'aut':
            prompt_template = "AUT Prompt:{}\nResponse:{}\nScore:\n"
        # This is format of trained models in Organisciak, Acar, Dumas, and Berthiaume
        return prompt_template.format(item, response)

    def _score_gpt(self, gptprompt, model='first', just_final=False):
        # gptprompt is the templated item+response. Use _craft_gptprompt. It can be a list of prompts.
        if model == 'first':
            model = self.models[0]
        response = openai.Completion.create(
            model=self._models[model],
            prompt=gptprompt,
            temperature=0,
            n=1,
            logprobs=None,
            stop='\n',
            max_tokens=1
        )
        if just_final:
            return [choice['text'] for choice in response.choices]
        else:
            return response