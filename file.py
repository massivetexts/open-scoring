# file.py
# Code for working with and reshaping various manner of entry files.
import pandas as pd

class WideData():
    '''
    Takes a file structured with Participant ID / GROUP / Prompt1 / Prompt2 / Prompt 3,
    and makes it long format
    '''
    
    def __init__(self, filename):
        self._original = pd.read_excel(filename).rename(columns={'Participant ID':'participant'})
        self._original.columns = self._original.columns.str.lower()
    
        self.df = self.make_long()
        
    def _clean_response(self, response):
        ''' Stripping punctuation, etc. '''
        clean = response.strip().strip('.').strip('!').strip(',').lower()
        replacements = [('/', ' '), ('door stop', 'doorstop'),
                        ('paper weight', 'paperweight')]
        for patt, sub in replacements:
            clean = clean.replace(patt, sub)
        return clean
    
    def make_long(self, wide=None, clean=True, drop_original=True):
        if not wide:
            wide=self._original
        # Melt to a participant / group / prompt / response_num df
        by_prompt = pd.melt(self._original, id_vars=['participant'],
                            var_name='prompt', value_name='responses')
        # Expand responses
        split_responses = pd.concat([by_prompt[['participant', 'prompt']], 
                                     by_prompt.responses.str.split('\n', expand=True)], 
                                    axis=1)
        # Make the expanded responses long
        df = (pd.melt(split_responses,
                      id_vars=['participant', 'prompt'], 
                      value_name='original_response', 
                      var_name='response_num')
              .sort_values(['participant','prompt', 'response_num'])
              .dropna()
             )
        
        if clean:
            df['response'] = df.original_response.apply(self._clean_response)
        else:
            df['response'] = df['original_response']
            
        df = df[df.response != '']
        
        if drop_original:
            df = df.drop(columns='original_response')
        
        return df
    
    def fluency(self, wide=False):
        fluency = (self.df.groupby(['participant', 'prompt'], as_index=False)[['response_num']]
                   .count()
                   .rename(columns={'response_num':'count'})
                  )
        if wide:
            fluency = fluency.pivot(index='participant', columns='prompt', values='count')
        return fluency
    
    def score(self, scorer, model, name=None, stop=False, idf=False, scorer_args={}):
        ''' Scores a full dataset of prompt/response columns. Those column names are expected.

        Provide an AUT_Scorer class, and a dict of arguments to pass to the scoring function.
        
        Adds a column of {name} to the internal data.df representation.

        e.g. 
        data = file.WideData('Measurement study/Participant level data/AlternateUses.xls')
        scorer = scoring.AUT_Scorer()
        scorer.load_model('EN_100_lsa', '/data/tasa/EN_100k.word2vec.bin')
        data.score(scorer, 'EN_100_lsa', idf=True)
        '''

        if not name:
            ''' Use model name as column name'''
            name = (model + ('_stop' if stop else '') + ('_idf' if idf else ''))
            
        if name in self.df.columns:
            print("Column %s already exists. Re-crunching and re-writing." % name)

        def scoring_func(x):
            y = scorer.originality(x['prompt'],
                                   x['response'],
                                   model=model, 
                                   stopword=stop, 
                                   term_weighting=idf,
                                   **scorer_args)
            return y

        self.df[name] = self.df.apply(scoring_func, axis=1)
        return None
                
    def score_all(self, scorer, idf=True, stop=True):
        ''' Score file with all models. This is a convenience function that expects
        each model to have the same settings, and uses a default column name'''
        
        for model in scorer.models:
            print("Scoring %s" % model)
            self.score(scorer, model, stop=stop, idf=idf)
                