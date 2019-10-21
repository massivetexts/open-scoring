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
    
    def make_long(self, wide=None, clean=True):
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
        return df
    
    def fluency(self, wide=False):
        fluency = (self.df.groupby(['participant', 'prompt'], as_index=False)[['response_num']]
                   .count()
                   .rename(columns={'response_num':'count'})
                  )
        if wide:
            fluency = fluency.pivot(index='participant', columns='prompt', values='count')
        return fluency