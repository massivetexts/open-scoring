# file.py
# Code for working with and reshaping various manner of entry files.
import pandas as pd

class WideData():
    '''
    Takes a file structured with Participant ID / GROUP / Prompt1 / Prompt2 / Prompt 3,
    and makes it long format
    '''
    
    def __init__(self, filename, id_cols=None):
        if filename.endswith('xls'):
            self._original = pd.read_excel(filename, convert_float=False).rename(columns={'Participant ID':'participant'})
        elif filename.endswith('csv'):
            self._original = pd.read_csv(filename).rename(columns={'Participant ID':'participant'})
        self._original.columns = self._original.columns.str.lower()

        if id_cols:
           self.id_cols = id_cols
        else:
            # Make a guess
            whitelist = ['participant', 'group']
            self.id_cols = [x for x in self._original.columns if x in whitelist]
    
        self.scored_columns = []
        self.df = self.to_long()
        
        
    def _clean_response(self, response):
        ''' Stripping punctuation, etc. '''
        clean = response.strip().strip('.').strip('!').strip(',').lower()
        replacements = [('/', ' '), ('door stop', 'doorstop'),
                        ('paper weight', 'paperweight')]
        for patt, sub in replacements:
            clean = clean.replace(patt, sub)
        return clean
    
    def to_long(self, df=None, clean=True, drop_original=True):
        if not df:
            df=self._original
        # Melt to a participant / group / prompt / response_num df
        by_prompt = pd.melt(self._original, id_vars=self.id_cols,
                            var_name='prompt', value_name='responses')
        # Expand responses
        split_responses = pd.concat([by_prompt[self.id_cols + ['prompt']], 
                                     by_prompt.responses.str.split('\n', expand=True)], 
                                    axis=1)
        # Make the expanded responses long
        df = (pd.melt(split_responses,
                      id_vars=self.id_cols + ['prompt'], 
                      value_name='original_response', 
                      var_name='response_num')
              .sort_values(self.id_cols + ['prompt', 'response_num'])
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
        fluency = (self.df.groupby(self.id_cols + ['prompt'], as_index=False)[['response_num']]
                   .count()
                   .rename(columns={'response_num':'count'})
                  )
        if wide:
            fluency = fluency.pivot_table(index=self.id_cols,
                                          columns='prompt',
                                          fill_value=0,
                                          values='count')
        return fluency
    
    def elaboration(self, wide=False):
        elab = (self.df.groupby(self.id_cols + ['prompt'], as_index=False)[['response_num']]
                   .count()
                   .rename(columns={'response_num':'count'})
                  )
        if wide:
            fluency = fluency.pivot_table(index=self.id_cols,
                                          columns='prompt',
                                          fill_value=0,
                                          values='count')
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
        if name not in self.scored_columns:
            self.scored_columns.append(name)
        return None
                
    def score_all(self, scorer, idf=True, stop=True):
        ''' Score file with all models. This is a convenience function that expects
        each model to have the same settings, and uses a default column name'''
        
        for model in scorer.models:
            print("Scoring %s" % model)
            self.score(scorer, model, stop=stop, idf=idf)
            
            
    def to_wide(self, aggfunc='mean'):
        ''' Convert scores back to a wide-format dataset'''
        if len(self.scored_columns):
            df = pd.pivot_table(self.df, index=self.id_cols, 
                                columns='prompt', values=self.scored_columns,
                                aggfunc=aggfunc)
            return df
        else:
            raise Exception("to_wide doesn't work before you've scored something!")
