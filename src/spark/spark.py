from collections import defaultdict

import pandas as pd
from pandas._typing import MergeHow

from .constants import PRIMARY_KEY
from .feat import Feat
from .inst import Inst


class SPARK:
    """
    A class representing the SPARK dataset.
    """

    #: A dictionary mapping instrument codes to their corresponding dataframes.
    instruments: dict[str, pd.DataFrame]

    def __init__(self, spark_pathname: str, instruments: list[Inst] = None):
        """
        Initializes the SPARK dataset with the specified instruments.

        :param spark_pathname: The SPARK data release directory, which should end with a date delimited by an underscore.
        :param instruments: A list of instrument names to include. If None, all instruments will be loaded.
        """
        self.instruments = Inst.get(spark_pathname, instruments)

    def join(self, features: list[Feat], how: MergeHow = "outer") -> pd.DataFrame:
        """
        Joins the specified features from the SPARK dataset into a single dataframe.

        :param features: A list of features to join.
        :param how: The type of join to perform. Refer to `pandas documentation <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html>`_ for more details.
        :return: A dataframe containing the joined features.
        """
        dfs = []

        groups = defaultdict(list)

        for feat in features:
            groups[feat.inst_code].append(feat)

        for inst_code, group in groups.items():
            inst_df = self.instruments[inst_code]
            cols_to_keep = [feat.source_col for feat in group] + [PRIMARY_KEY]
            inst_df = inst_df[cols_to_keep].set_index(PRIMARY_KEY)

            inst_df = inst_df.rename(
                columns={feat.source_col: feat.col for feat in group}
            )
            dfs.append(inst_df)

        df = dfs[0].join(dfs[1:], how=how)

        return df
