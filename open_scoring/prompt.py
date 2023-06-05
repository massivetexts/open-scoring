import os
from typing import Optional, Union, List
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate, load_prompt
from langchain.schema import HumanMessage
import langchain

import importlib
from pathlib import Path
import csv
import re
import time
from io import StringIO

import logging

import pandas as pd
from sklearn.model_selection import train_test_split
import hashlib
import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

ASSETS_PATH = importlib.resources.files(__name__.split('.')[0]) / 'assets'
PROMPTS_PATH = ASSETS_PATH / 'prompts'

class ScoredCSVOutputParser():
    """Parse out multiple line comma separated lists."""
    def __init__(self, examples: Optional[str] = ['response', 'score']):
        self.examples = examples

    def get_format_instructions(self) -> str:
        return (
            "Your response should be a list of comma separated values, "
            f"eg: `{','.join(self.examples)}`"
        )
    
    def parse(self, text: str) -> List[List[str]]:
        """Parse the output of an LLM call."""
        csv_file = StringIO(text)
        cleaned = []
        for row in csv.reader(csv_file):
            if len(row) > 2:
                # fallback on re
                line = ",".join(row)
                row = re.split(', ?(\d\.\d+)', line)[:-1]
            assert len(row) == 2, f"Currently assumes only two fields: response, score; seeing {len(row)}"
            response, score = row
            response = response.strip('-').strip()
            try:
                score = float(score.strip())
            except ValueError:
                score = score.strip()
            cleaned.append((response, score))
        return cleaned
    
class NumberedListOutputParser():
    """Parse out a numbered list."""
    def __init__(self, examples: Optional[str] = ['response', 'score']):
        self.examples = examples

    def get_format_instructions(self) -> str:
        return "" # no format instructions - it should be apparent from the template
    
    def parse(self, text: str) -> List[List[str]]:
        """Parse the output of an LLM call."""
        lines = text.strip().split('\n')
        outputs = []
        for line in lines:
            value = re.split('^\d+\. ?', line)[-1]
            outputs.append(value)
        return outputs

class PromptScorer:
    def __init__(self,
                 openai_key_path: Optional[str] = None,
                 model: str = 'gpt-3.5-turbo'
                 ):
        if openai_key_path:
            with open(openai_key_path) as f:
                openai_api_key = f.readline().strip()
            os.environ["OPENAI_API_KEY"] = openai_api_key
        self.model = model
        self.chat = ChatOpenAI(temperature=0, model_name=model, request_timeout=240)
        self.parser = ScoredCSVOutputParser()

    def originality(self,
                    response: Union[str, List[str]],
                    prompt: Optional[str] = None,
                    full_prompt: Optional[str] = None,
                    
                    task: str = 'uses',
                    examples: Optional[List[dict]] = None,
                    return_prompt: Optional[bool] = False,
                    style: Union[int, PromptTemplate] = 1) -> float:
        
        results = self._generic_scorer(
            'originality', response=response, prompt=prompt, full_prompt=full_prompt,
            task=task, examples=examples, return_prompt=return_prompt, style=style
        )
        return results
    
    def flexibility(self,
                    response: Union[str, List[str]],
                    prompt: Optional[str] = None,
                    full_prompt: Optional[str] = None,
                    
                    task: str = 'uses',
                    examples: Optional[List[dict]] = None,
                    style: Union[int, PromptTemplate] = 1) -> float:
        if (type(response) is list and len(response) <= 1) or (type(response) is str):
            logging.debug('Only one response given - assigning max flexibility')
            # should return null maybe? Like dividing by zero...
            return 5
        
        results = self._generic_scorer(
            'flexibility', response=response, prompt=prompt, full_prompt=full_prompt,
            task=task, examples=examples, style=style
        )
        return results
    

    def summarizer(self,
                prompt: Optional[str] = None,
                full_prompt: Optional[str] = None,
                temperature: float = 0,
                task: str = 'uses',
                examples: Optional[List[dict]] = None,
                style: Union[int, PromptTemplate] = 1) -> float:
        ''' Special construct that takes few show examples and tries to summarize them into a codebook, or a description.

        Intended for use with a great deal of examples.

        If task == 'uses_mixed', the prompt is written to allow for a variety of task prompts, and the examples list should be prompt,response,score, while the prompt and full_prompt should be blank. Else, the examples are response/score
        
        .'''
        results = self._generic_scorer(
            'summarizer', response=None, prompt=prompt, full_prompt=full_prompt,
            task=task, examples=examples, parse=False, style=style
        )
        return results

    def _generic_scorer(self, 
                        construct: str,
                        response: Union[str, List[str], None],
                        prompt: Optional[str] = None,
                        full_prompt: Optional[str] = None,
                        temperature: float = None,
                        task: str = 'uses',
                        parse: bool = True,
                        examples: Optional[List[dict]] = None,
                        return_prompt: Optional[bool] = None,
                        style: Union[str, int, PromptTemplate] = 1) -> float:
        
        if not prompt and not full_prompt and 'mixed' not in task:
            raise ValueError("Either prompt or full_prompt must be provided, unless using 'mixed' task types (e.g. 'uses_mixed')")

        if task == 'custom' and not full_prompt:
            full_prompt = prompt

        if isinstance(style, int) or isinstance(style, str):
            prompt_file = PROMPTS_PATH / f'{construct}/{task}/{style}.yaml'
            prompt_template = load_prompt(prompt_file)
        else:
            prompt_template = PromptTemplate(full_prompt)
        
        

        template_args = dict()
        parser = self.parser

        if 'splitlist' in style:
            # submit examples and example scores separately
            template_args['example_uses'] = "\n".join([f'{i+1}. {ex["response"]}' for i, ex in enumerate(examples)])
            template_args['example_scores'] = "\n".join([f'{i+1}. {ex["score"]}' for i, ex in enumerate(examples)])
            if type(response) is str:
                response = [response]
            template_args['responses'] = "\n".join([f'{i+1+len(examples)}. {response}' for i, (response) in enumerate(response)])
            parser = NumberedListOutputParser()
        elif examples:
            prompt_template.examples = examples
        
        if prompt is not None:
            template_args['item'] = prompt.upper()

        if (response is not None) and ('splitlist' not in style):
            response_str = self.response_formatter(response)
            template_args['responses'] = response_str
        if parse:
            if 'splitlist' not in style:
                template_args['format_instructions'] = self.parser.get_format_instructions()
        if len(template_args) == 0:
            # langchain doesn't support no args, so those templates take a 'blank'
            template_args['blank'] = ''
        msg = prompt_template.format(**template_args).strip()
        if return_prompt:
            return msg
        logging.debug(msg)
        if temperature:
            self.chat.temperature = temperature
        results = self.chat([HumanMessage(content=msg)])
        if temperature:
            self.chat.temperature = 0
        if parse:
            try:
                parsed_results = parser.parse(results.content)
                if 'splitlist' in style:
                    parsed_results = list(zip(response, parsed_results))
                return parsed_results
            except:
                print("temporarily catching error and returning raw output")
                return results
        else:
            return results.content
        
    def response_formatter(self,
                           response: Union[str, List[str]],
                           number_response: bool = False):
        '''Take a list of originality string responses (or a single string) and format input list'''
        if type(response) is list:
            response_str = ''
            for i in range(1, len(response)+1):
                prefix = f"{i}. " if number_response else "- "
                response_str += f'{prefix}{response[i-1]}\n'
            response_str = response_str.strip('\n')
        elif type(response) is str:
            response_str = f"1. {response}"

        return response_str


class DatasetPromptScorer():
    """
    A class that takes a dataframe of originality test responses, splits into test/val/train, and scores using a PromptScorer.

    Attributes
    ----------
    train_prop : float
        The proportion of data to be used for training.
    val_prop : float
        The proportion of data to be used for validation.
    test_prop : float
        The proportion of data to be used for testing.
    seed : int
        The seed for the random number generator for reproducibility.
    type_col : str
        The name of the type column in the input data.
    prompt_col : str
        The name of the prompt column in the input data.
    id_col : str
        The name of the id column in the input data.
    response_col : str
        The name of the response column in the input data.
    score_col : str
        The name of the score column in the input data.
    traindata : DataFrame
        The data to be used for training.
    valdata : DataFrame
        The data to be used for validation.
    testdata : DataFrame
        The data to be used for testing.
    """


    def __init__(self, data, train_prop=0.8, val_prop=0.05, seed=123,
        type_col='type', testtype=None, prompt_col='prompt', prompt=None, id_col='id', response_col='response', score_col='target'):
        """
        Initializes the class and splits the input data into train, validation, and test datasets.

        If there's a split column, use that for train/val/test labels.
        """
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.test_prop = 1 - train_prop - val_prop

        self.seed = seed

        self.type_col = type_col
        self.prompt_col = prompt_col
        self.id_col = id_col
        self.response_col = response_col
        self.score_col = score_col

        if (self.type_col not in data):
            if not testtype:
                raise KeyError(f"If you don't have a {self.type_col} column in your data, you need to specify your testtype (for the whole dataset) with the `testtype` arg")
            data[type_col] = testtype

        if (self.prompt_col not in data):
            if not prompt:
                raise KeyError(f"If you don't have a {self.prompt_col} column in your data, you need to specify your prompt (for the whole dataset) with the `prompt` arg")
            data[type_col] = testtype

        for col in [self.response_col, self.score_col]:
            if col not in data.columns:
                raise KeyError(f"Your input data needs columns specifying testtype, prompt (the item), id (a unique id), response, and score (for the train data). Missing '{col}' column.")
        
        if self.id_col not in data:
            def compute_hash(row):
                return hashlib.sha256(row.encode()).hexdigest()
            data[id_col] = (data[type_col] + data[prompt_col] + data[response_col]).apply(compute_hash)


        assert self.test_prop > 0, "Test/val need to be proportions that add to <1"
        
        if 'split' in data.columns:
            self.traindata = data[data.split == 'train']
            self.valdata = data[data.split == 'val']
            self.testdata = data[data.split == 'test']

            self.train_prop = len(self.traindata)/len(data)
            self.val_prop = len(self.valdata)/len(data)
            self.test_prop = len(self.testdata)/len(data)
        else:
            self.traindata, self.valdata, self.testdata = self.split_data(data)

    def split_data(self, data):
        """
        Splits the input data into train, validation, and test datasets.

        Parameters
        ----------
        data : DataFrame
            The input data to be split.

        Returns
        -------
        train, val, test : tuple of DataFrame
            The train, validation, and test datasets.
        """
         
        train_val, test = train_test_split(data, test_size=1-self.train_prop, random_state=self.seed)
        relative_val_prop = self.val_prop / (1 - (1 - self.train_prop))
        train, val = train_test_split(train_val, test_size=relative_val_prop, random_state=self.seed)
        return train, val, test

    def score(self, name=None, style='1', model='gpt-3.5-turbo', n_examples=5, n_per_prompt=5,
              repeat = 1, return_errors=False, save_location=None, fifty_scale=False, return_prompt=False,
              example_strategy='random'):
        """
        Scores the test data. Input dataframe needs an id column, a response column, and a score column.

        Parameters
        ----------
        name_prefix : str
            The name for this run.
        style : str, optional
            The style for the scorer (default is '1').
        model : str, optional
            The model to be used (default is 'gpt-3.5-turbo').
        n_examples : int, optional
            The number of examples to be used (default is 5).
        n_per_prompt : int, optional
            The number of responses to score per LLM-call (default is 5).
        repeat : int, optional
            The number of times to repeat the scoring (default is 1).
        return_errors: bool, optional
            Whether to return a tuple of unparsable_responses, correct_responses. (default is False)
        save_location : str, optional
            The directory to save outputs to. Filename is auto-named (default is None).
        example_strategy : str, optional
            The strategy for the examples (default is 'random').
        fifty_scale: bool, optional
            Whether to use a 10-50 scale. Ensure that the prompt matches.
        return_prompt: bool, optional
            For Debugging, just return the prompts without running (default is False)

        Returns
        -------
        DataFrame
            The results of the scoring.
        """

        scorer = PromptScorer(model=model) #needs openaikeypath in path

        ticks_per_run = (self.testdata.groupby(['type', 'prompt']).size() / n_per_prompt).apply(np.ceil).sum()
        total_ticks = int(ticks_per_run * repeat)

        pbar = tqdm(total=total_ticks)

        all_results = []
        all_errs = []
        timestamp = time.time()
        fname = f"{name.replace('-','')}-{model}-{n_examples}-{n_per_prompt}-{example_strategy}-{style}-{timestamp}.csv"
        prompts = []

        for run_n in range(repeat):
            for (testtype, prompt), type_subset in self.testdata.groupby(['type', 'prompt']):
                chunks = self._split_dataframe(type_subset, n_per_prompt)
                for i, chunk in enumerate(chunks):
                    pbar.set_description(f"Run {run_n+1}: Processing {testtype}/{prompt} - chunk {i+1}")
                    to_score = chunk.response.str.strip().tolist()
                    example_pool = self.traindata[self.traindata[self.prompt_col] == prompt]
                    examples_dict, example_ids = self.get_examples(n_examples, example_pool, example_strategy)

                    if fifty_scale:
                        new_examples_dict = []
                        for example in examples_dict:
                            example['score'] = int(example['score']*10)
                            new_examples_dict.append(example)
                        examples_dict = new_examples_dict
                    # Score
                    try:
                        results = scorer.originality(prompt=prompt, response=to_score,
                                            examples=examples_dict, return_prompt=return_prompt,
                                            task=testtype, style=style)
                        if return_prompt:
                            prompts.append(results)
                            pbar.update(1)
                            continue
                    except KeyboardInterrupt:
                        raise
                    except:
                        ## TODO catch errors, especially API timeouts (wait 20s and try again)
                        raise

                    if type(results) is langchain.schema.AIMessage:
                        # this is only returned if there's a parsing error.
                        all_errs.append(results)
                        continue
            
                    # Parse output
                    results_df = pd.DataFrame([(testtype, prompt, response, prediction) for response, prediction in results],
                                columns=[self.type_col, self.prompt_col, self.response_col, 'prediction'])
                    # join with full data
                    merged_df = chunk.merge(results_df)
                    if len(merged_df) != len(chunk):
                        if len(results_df) == len(chunk):
                            logging.debug('Join failed - trying simply to align columns')
                            merged_df = chunk.copy()
                            merged_df[['response_check', 'prediction']] = results
                        else:
                            logging.warning("Second fallback - left join - some scores may be missed")
                            merged_df = chunk.merge(results_df, how='left')

                    if fifty_scale:
                        def divide_if_possible(x):
                            try:
                                y = int(x)
                                return y/10
                            except:
                                return x
                        merged_df['prediction'] =merged_df['prediction'].apply(divide_if_possible)
                    # add metadata
                    merged_df['example_ids'] = example_ids
                    merged_df['run_n'] = run_n
                    merged_df['n_examples'] = n_examples
                    merged_df['n_per_prompt'] = n_per_prompt
                    merged_df['model'] = model
                    merged_df['example_strategy'] = example_strategy
                    merged_df['name'] = name
                    merged_df['style'] = style
                    merged_df['timestamp'] = timestamp

                    pbar.update(1)
                    all_results.append(merged_df)
                    # save intermediate state, in run is cut short
                    if save_location:
                        pd.concat(all_results).to_csv(Path(save_location) / fname, index=False)

        if return_prompt and not return_errors:
            return prompts
        elif return_prompt:
            return all_errs, prompts
        
        all_results_df = pd.concat(all_results)
        if save_location:
            all_results_df.to_csv(Path(save_location) / fname, index=False)

        if return_errors:
            return all_errs, all_results_df
        else:
            return all_results_df

    def get_examples(self, n, data=None, strategy='random', include_prompt=False):
        """
        Gets examples from the training data based on a specified strategy.

        Parameters
        ----------
        n : int
            The number of examples to get.
        strategy : str, optional
            The strategy for getting the examples (default is 'random').

        Returns
        -------
        examples_dict, example_ids : tuple where the first item is a list of examples, and the second is a reference of which ids are used
        """
        if type(data) == type(None):
            data = self.traindata
        if strategy == 'random':
            n = min(n, len(data))
            examples = data.sample(n)[[self.id_col, self.prompt_col, self.response_col, self.score_col]]
        else:
            raise ValueError(f'strategy="{strategy}" is not implemented')
        if include_prompt:
            examples_dict = examples[[self.prompt_col, self.response_col, self.score_col]].rename(columns={self.prompt_col: 'item', self.response_col:'response', self.score_col:'score'}).to_dict(orient='records')
        else:
            examples_dict = examples[[self.response_col, self.score_col]].rename(columns={self.response_col:'response', self.score_col:'score'}).to_dict(orient='records')
        example_ids = ",".join(examples[self.id_col].tolist())
        return examples_dict, example_ids
    
    def _split_dataframe(self, df, chunk_size):
        ''' Split into chunks, for iteration'''
        chunks = []
        num_chunks = len(df) // chunk_size + 1

        for i in range(num_chunks):
            chunks.append(df[i*chunk_size:(i+1)*chunk_size])

        return chunks
       