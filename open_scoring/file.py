# file.py
# Code for working with and reshaping various manner of entry files.
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()


class WideData:
    """
    Takes a file structured with Participant ID / GROUP / Prompt1 / Prompt2 / Prompt 3,
    and makes it long format

    id_cols: columns that id the respondant. If left to the default None, makes a guess by
            looking for a column named 'participant' or 'group' (or both)

    participant_id: name of participant column. This will be renamed to 'participant' internally.

    response_aggfunc: the default way of aggregating multiple responses in wide format.
    """

    def __init__(
        self,
        filename,
        id_cols=None,
        participant_id="Participant ID",
        response_aggfunc="mean",
    ):
        self.response_aggfunc = response_aggfunc

        if filename.endswith("xls") or filename.endswith("xlsx"):
            self._original = pd.read_excel(filename, convert_float=False).rename(
                columns={participant_id: "participant"}
            )
        elif filename.endswith("csv"):
            self._original = pd.read_csv(filename).rename(
                columns={participant_id: "participant"}
            )
        self._original.columns = self._original.columns.str.lower()

        if id_cols:
            self.id_cols = id_cols
        else:
            # Make a guess
            whitelist = ["participant", "group"]
            self.id_cols = [x for x in self._original.columns if x in whitelist]

        self.scored_columns = []
        self.df = self.to_long()

    def _clean_response(self, response):
        """Stripping punctuation, etc."""
        clean = response.strip().strip(".").strip("!").strip(",").lower()
        replacements = [
            ("/", " "),
            ("door stop", "doorstop"),
            ("paper weight", "paperweight"),
        ]
        for patt, sub in replacements:
            clean = clean.replace(patt, sub)
        return clean

    def to_long(self, df=None, clean=True, drop_original=True):
        if not df:
            df = self._original
        # Melt to a participant / group / prompt / response_num df
        by_prompt = pd.melt(
            self._original,
            id_vars=self.id_cols,
            var_name="prompt",
            value_name="responses",
        )
        # Expand responses
        split_responses = pd.concat(
            [
                by_prompt[self.id_cols + ["prompt"]],
                by_prompt.responses.str.split("\n", expand=True),
            ],
            axis=1,
        )
        # Make the expanded responses long
        df = (
            pd.melt(
                split_responses,
                id_vars=self.id_cols + ["prompt"],
                value_name="original_response",
                var_name="response_num",
            )
            .sort_values(self.id_cols + ["prompt", "response_num"])
            .dropna()
        )

        if clean:
            df["response"] = df.original_response.apply(self._clean_response)
        else:
            df["response"] = df["original_response"]

        df = df[df.response != ""]

        if drop_original:
            df = df.drop(columns="original_response")

        return df

    def fluency(self, wide=False):
        fluency = (
            self.df.groupby(self.id_cols + ["prompt"], as_index=False)[["response_num"]]
            .count()
            .rename(columns={"response_num": "count"})
        )
        if wide:
            fluency = fluency.pivot_table(
                index=self.id_cols, columns="prompt", fill_value=0, values="count"
            )
        return fluency

    def elaboration(self, scorer, wide=False, elabfunc="whitespace", aggfunc="default"):
        """Adds an elaboration column to the internal representation and returns it.

        Needs scorer simply because that's where IDF dict and spacy is initialized.
        elabfunc is the strategy for calculating elaboration. Read the scorer docs for details.
        aggfunc is how multiple responses are aggregated.
        """

        self.df["elaboration"] = self.df.response.apply(
            lambda x: scorer.elaboration(x, elabfunc=elabfunc)
        )

        if aggfunc == "default":
            aggfunc = self.response_aggfunc

        df = self.df[self.id_cols + ["prompt", "elaboration"]]
        if wide:
            df = df.pivot_table(index=self.id_cols, columns="prompt", aggfunc=aggfunc)

        return df

    def score(
        self,
        scorer,
        model,
        name=None,
        stop=False,
        idf=False,
        exclude_target=False,
        alt_prompt=None,
        scorer_args={},
    ):
        """
        Scores a full dataset of prompt/response columns. Those column names are expected.

        Provide an AUT_Scorer class, and a dict of arguments to pass to the scoring function.

        Adds a column of {name} to the internal data.df representation.

        e.g.
        data = file.WideData('Measurement study/Participant level data/AlternateUses.xls')
        scorer = scoring.AUT_Scorer()
        scorer.load_model('EN_100_lsa', '/data/tasa/EN_100k.word2vec.bin')
        data.score(scorer, 'EN_100_lsa', idf=True)
        """

        if not name:
            """Use model name as column name"""
            name = model + ("_stop" if stop else "") + ("_idf" if idf else "")

        if name in self.df.columns:
            print("Column %s already exists. Re-crunching and re-writing." % name)

        if hasattr(scorer, "originality_batch"):
            n = len(self.df)
            targets = self.df["prompt"].tolist() if not alt_prompt else [alt_prompt] * n
            responses = self.df["response"].tolist()
            self.df[name] = scorer.originality_batch(
                targets,
                responses,
                model=model,
                stopword=stop,
                term_weighting=idf,
                exclude_target=exclude_target,
                **scorer_args
            )
        else:

            def scoring_func(x):
                prompt = x["prompt"] if not alt_prompt else alt_prompt
                y = scorer.originality(
                    prompt,
                    x["response"],
                    model=model,
                    stopword=stop,
                    term_weighting=idf,
                    exclude_target=exclude_target,
                    **scorer_args
                )
                return y

            self.df[name] = self.df.progress_apply(scoring_func, axis=1)

        if name not in self.scored_columns:
            self.scored_columns.append(name)
        return None

    def score_all(self, scorer, idf=True, stop=True, exclude_target=False):
        """Score file with all models. This is a convenience function that expects
        each model to have the same settings, and uses a default column name"""

        for model in scorer.models:
            print("Scoring %s" % model)
            self.score(scorer, model, stop=stop, idf=idf, exclude_target=exclude_target)

    def to_wide(self, aggfunc="default"):
        """Convert scores back to a wide-format dataset.

        aggfunc: how multiple responses for the same prompt are aggregated.
        Default is 'mean', other sensible options are 'min' and 'max'. A function
        can be passed.
        """
        if aggfunc == "default":
            aggfunc = self.response_aggfunc

        if len(self.scored_columns):
            df = pd.pivot_table(
                self.df,
                index=self.id_cols,
                columns="prompt",
                values=self.scored_columns,
                aggfunc=aggfunc,
            )
            return df
        else:
            raise Exception("to_wide doesn't work before you've scored something!")
