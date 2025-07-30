from collections import defaultdict

import pandas as pd
from pandas._typing import MergeHow

from .constants import PRIMARY_KEY
from .inst import Inst, Feat


class SPARK:
    """
    A class for reading and manipulating the instruments of the SPARK dataset.

    #. Begin by intialising the class with the path to the SPARK dataset directory and the instruments you would like to manipulate.
    #. Then you can construct dataframes of features indexed by the primary key using the :meth:`~spark.spark.SPARK.join` method.

    .. code-block:: python

        from spark import SPARK, Inst, Feat

        ds = SPARK(
            spark_pathname=spark_pathname,
            instruments=[Inst.RBSR],
        )

        df = ds.join(features: [Feat.RBSR_TOTAL_FINAL_SCORE])
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

    def join(
        self, features: list[Feat], how: MergeHow = "outer", rename: bool = True
    ) -> pd.DataFrame:
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

            if rename:
                inst_df = inst_df.rename(
                    columns={feat.source_col: feat.col for feat in group}
                )

            dfs.append(inst_df)

        df = dfs[0].join(dfs[1:], how=how)

        return df

    @staticmethod
    def init_and_join(
        spark_pathname: str, features: list[Feat], how: MergeHow = "outer"
    ) -> tuple["SPARK", pd.DataFrame, list[Inst]]:
        """
        Initializes the SPARK dataset and joins the specified features into a dataframe.

        This is a convenience method that combines the initialization of the SPARK dataset with the joining of features, preventing mismatches between requested features and the instruments loaded.

        :param spark_pathname: The SPARK data release directory pathname.
        :param features: The features to join.
        :param how: The type of join to perform. Refer to `pandas documentation <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html>`_ for more details.
        :return: A tuple containing the SPARK dataset instance, the joined dataframe, and a list of instruments used in the join.
        """
        instruments: list[Inst] = [
            Inst.from_code(inst_code)
            for inst_code in set(feat.inst_code for feat in features)
        ]

        ds = SPARK(
            spark_pathname=spark_pathname,
            instruments=instruments,
        )

        df = ds.join(features=features, how=how)

        return ds, df, instruments
