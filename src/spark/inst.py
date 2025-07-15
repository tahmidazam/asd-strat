from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from .feat import Feat


class Inst(Enum):
    """
    A SPARK instrument.
    """

    @property
    def code(self) -> str:
        """
        The instrument code, a unique identifier of the instrument.

        :return: The instrument code.
        """
        return self.value[0].upper()

    @property
    def base_filepath(self) -> str:
        """
        The base filepath of the instrument, which is used to construct the full filepath.

        The base filepath may contain a directory.

        :return: The base filepath of the instrument.
        """
        return self.value[1]

    @property
    def question_features(self) -> Optional[list[Feat]]:
        """
        The question features of the instrument.

        :return: A list of question features, or None if the instrument does not have question features.
        """
        return self.value[2]

    @property
    def subscale_score_features(self) -> Optional[list[Feat]]:
        """
        The subscale score features of the instrument.

        :return: A list of subscale score features, or None if the instrument does not have subscale score features.
        """
        return self.value[3]

    @property
    def final_score_feature(self) -> Optional[Feat]:
        """
        The final score feature of the instrument.

        :return: The final score feature, or None if the instrument does not have a final score feature.
        """
        return self.value[4]

    def get_filepath(self, spark_pathname: str) -> Path:
        """
        Generates the filepath for an instrument using the provided SPARK data release directory.

        The date string is extracted from the SPARK pathname, and is joined with the instrument's base filepath
        separated by a hyphen.

        :param spark_pathname: The SPARK data release directory, which should end with a date delimited by an underscore.
        :return: The filepath for the instrument.
        """
        date_string = spark_pathname.split("_")[-1]
        spark_path = Path(spark_pathname)
        inst_filename = f"{self.base_filepath}-{date_string}.csv"
        filepath = spark_path / inst_filename
        return filepath

    def read_csv(self, spark_pathname: str):
        """
        Reads the instrument from disk given the SPARK data release directory.

        :param spark_pathname: The SPARK data release pathname.
        :return: A dataframe containing the instrument data.
        """
        filepath = self.get_filepath(spark_pathname)
        return pd.read_csv(filepath, engine="pyarrow")

    @staticmethod
    def get(
        spark_pathname: str, instruments: list["Inst"] = None
    ) -> dict[str, pd.DataFrame]:
        """
        Reads the specified instruments from disk given the SPARK data release directory.

        :param spark_pathname: The SPARK data release pathname.
        :param instruments: A list of instruments to read. If None, all instruments will be read.
        :return: A dictionary mapping instrument codes to their respective dataframes.
        """
        inst_iter = tqdm(instruments or Inst, desc="Reading instruments")

        return {
            inst.code: inst.read_csv(spark_pathname=spark_pathname)
            for inst in inst_iter
        }

    DCDQ = (
        "dcdq",
        "dcdq",
        [
            Feat.DCDQ_Q01_THROW_BALL,
            Feat.DCDQ_Q02_CATCH_BALL,
            Feat.DCDQ_Q03_HIT_BALL,
            Feat.DCDQ_Q04_JUMP_OBSTACLES,
            Feat.DCDQ_Q05_RUN_FAST_SIMILAR,
            Feat.DCDQ_Q06_PLAN_MOTOR_ACTIVITY,
            Feat.DCDQ_Q07_PRINTING_WRITING_DRAWING_FAST,
            Feat.DCDQ_Q08_PRINTING_LETTERS_LEGIBLE,
            Feat.DCDQ_Q09_APPROPRIATE_TENSION_PRINTING_WRITING,
            Feat.DCDQ_Q10_CUTS_PICTURES_SHAPES,
            Feat.DCDQ_Q11_LIKES_SPORTS_MOTORS_SKILLS,
            Feat.DCDQ_Q12_LEARNS_NEW_MOTOR_TASKS,
            Feat.DCDQ_Q13_QUICK_COMPETENT_TIDYING_UP,
            Feat.DCDQ_Q14_BULL_IN_CHINA_SHOP,
            Feat.DCDQ_Q15_FATIGUE_EASILY,
        ],
        [
            Feat.DCDQ_CONTROL_DURING_MOVEMENT,
            Feat.DCDQ_FINE_MOTOR_HANDWRITING,
            Feat.DCDQ_GENERAL_COORDINATION,
        ],
        Feat.DCDQ_FINAL_SCORE,
    )

    RBSR = (
        "rbsr",
        "rbsr",
        [
            Feat.RBSR_Q01_WHOLE_BODY,
            Feat.RBSR_Q02_HEAD,
            Feat.RBSR_Q03_HAND_FINGER,
            Feat.RBSR_Q04_LOCOMOTION,
            Feat.RBSR_Q05_OBJECT_USAGE,
            Feat.RBSR_Q06_SENSORY,
            Feat.RBSR_Q07_HITS_SELF_BODY,
            Feat.RBSR_Q08_HITS_SELF_AGAINST_OBJECT,
            Feat.RBSR_Q09_HITS_SELF_WITH_OBJECT,
            Feat.RBSR_Q10_BITES_SELF,
            Feat.RBSR_Q11_PULLS,
            Feat.RBSR_Q12_RUBS,
            Feat.RBSR_Q13_INSERTS_FINGER,
            Feat.RBSR_Q14_SKIN_PICKING,
            Feat.RBSR_Q15_ARRANGING,
            Feat.RBSR_Q16_COMPLETE,
            Feat.RBSR_Q17_WASHING,
            Feat.RBSR_Q18_CHECKING,
            Feat.RBSR_Q19_COUNTING,
            Feat.RBSR_Q20_HOARDING,
            Feat.RBSR_Q21_REPEATING,
            Feat.RBSR_Q22_TOUCH_TAP,
            Feat.RBSR_Q23_EATING,
            Feat.RBSR_Q24_SLEEP,
            Feat.RBSR_Q25_SELF_CARE,
            Feat.RBSR_Q26_TRAVEL,
            Feat.RBSR_Q27_PLAY,
            Feat.RBSR_Q28_COMMUNICATION,
            Feat.RBSR_Q29_THINGS_SAME_PLACE,
            Feat.RBSR_Q30_OBJECTS,
            Feat.RBSR_Q31_BECOMES_UPSET,
            Feat.RBSR_Q32_INSISTS_WALKING,
            Feat.RBSR_Q33_INSISTS_SITTING,
            Feat.RBSR_Q34_DISLIKES_CHANGES,
            Feat.RBSR_Q35_INSISTS_DOOR,
            Feat.RBSR_Q36_LIKES_PIECE_MUSIC,
            Feat.RBSR_Q37_RESISTS_CHANGE,
            Feat.RBSR_Q38_INSISTS_ROUTINE,
            Feat.RBSR_Q39_INSISTS_TIME,
            Feat.RBSR_Q40_FASCINATION_SUBJECT,
            Feat.RBSR_Q41_STRONGLY_ATTACHED,
            Feat.RBSR_Q42_PREOCCUPATION,
            Feat.RBSR_Q43_FASCINATION_MOVEMENT,
        ],
        [
            Feat.RBSR_I_STEREOTYPED_BEHAVIOR_SCORE,
            Feat.RBSR_II_SELF_INJURIOUS_SCORE,
            Feat.RBSR_III_COMPULSIVE_BEHAVIOR_SCORE,
            Feat.RBSR_IV_RITUALISTIC_BEHAVIOR_SCORE,
            Feat.RBSR_V_SAMENESS_BEHAVIOR_SCORE,
            Feat.RBSR_VI_RESTRICTED_BEHAVIOR_SCORE,
        ],
        Feat.RBSR_TOTAL_FINAL_SCORE,
    )

    SCQ = (
        "scq",
        "scq",
        [
            Feat.SCQ_Q01_PHRASES,
            Feat.SCQ_Q02_CONVERSATION,
            Feat.SCQ_Q03_ODD_PHRASE,
            Feat.SCQ_Q04_INAPPROPRIATE_QUESTION,
            Feat.SCQ_Q05_PRONOUNS_MIXED,
            Feat.SCQ_Q06_INVENTED_WORDS,
            Feat.SCQ_Q07_SAME_OVER,
            Feat.SCQ_Q08_PARTICULAR_WAY,
            Feat.SCQ_Q09_EXPRESSIONS_APPROPRIATE,
            Feat.SCQ_Q10_HAND_TOOL,
            Feat.SCQ_Q11_INTEREST_PREOCCUPY,
            Feat.SCQ_Q12_PARTS_OBJECT,
            Feat.SCQ_Q13_INTERESTS_INTENSITY,
            Feat.SCQ_Q14_SENSES,
            Feat.SCQ_Q15_ODD_WAYS,
            Feat.SCQ_Q16_COMPLICATED_MOVEMENTS,
            Feat.SCQ_Q17_INJURED_DELIBERATELY,
            Feat.SCQ_Q18_OBJECTS_CARRY,
            Feat.SCQ_Q19_BEST_FRIEND,
            Feat.SCQ_Q20_TALK_FRIENDLY,
            Feat.SCQ_Q21_COPY_YOU,
            Feat.SCQ_Q22_POINT_THINGS,
            Feat.SCQ_Q23_GESTURES_WANTED,
            Feat.SCQ_Q24_NOD_HEAD,
            Feat.SCQ_Q25_SHAKE_HEAD,
            Feat.SCQ_Q26_LOOK_DIRECTLY,
            Feat.SCQ_Q27_SMILE_BACK,
            Feat.SCQ_Q28_THINGS_INTERESTED,
            Feat.SCQ_Q29_SHARE,
            Feat.SCQ_Q30_JOIN_ENJOYMENT,
            Feat.SCQ_Q31_COMFORT,
            Feat.SCQ_Q32_HELP_ATTENTION,
            Feat.SCQ_Q33_RANGE_EXPRESSIONS,
            Feat.SCQ_Q34_COPY_ACTIONS,
            Feat.SCQ_Q35_MAKE_BELIEVE,
            Feat.SCQ_Q36_SAME_AGE,
            Feat.SCQ_Q37_RESPOND_POSITIVELY,
            Feat.SCQ_Q38_PAY_ATTENTION,
            Feat.SCQ_Q39_IMAGINATIVE_GAMES,
            Feat.SCQ_Q40_COOPERATIVELY_GAMES,
        ],
        None,
        Feat.SCQ_FINAL_SCORE,
    )

    CBCL_6_18 = (
        "cbcl_6_18",
        "cbcl_6_18",
        [
            Feat.CBCL_6_18_Q001_ACTS_YOUNG,
            Feat.CBCL_6_18_Q002_DRINKS_ALCOHOL,
            Feat.CBCL_6_18_Q003_ARGUES,
            Feat.CBCL_6_18_Q004_FAILS_TO_FINISH,
            Feat.CBCL_6_18_Q005_VERY_LITTLE_ENJOYMENT,
            Feat.CBCL_6_18_Q006_BOWEL_MOVEMENTS_OUTSIDE,
            Feat.CBCL_6_18_Q007_BRAG_BOAST,
            Feat.CBCL_6_18_Q008_CONCENTRATE,
            Feat.CBCL_6_18_Q009_OBSESSIONS,
            Feat.CBCL_6_18_Q010_RESTLESS,
            Feat.CBCL_6_18_Q011_TOO_DEPENDENT,
            Feat.CBCL_6_18_Q012_LONELINESS,
            Feat.CBCL_6_18_Q013_CONFUSED,
            Feat.CBCL_6_18_Q014_CRIES_A_LOT,
            Feat.CBCL_6_18_Q015_CRUELTY_ANIMALS,
            Feat.CBCL_6_18_Q016_CRUELTY_OTHERS,
            Feat.CBCL_6_18_Q017_DAYDREAMS,
            Feat.CBCL_6_18_Q018_HARMS_SELF,
            Feat.CBCL_6_18_Q019_DEMANDS_ATTENTION,
            Feat.CBCL_6_18_Q020_DESTROYS_OWN_THINGS,
            Feat.CBCL_6_18_Q021_DESTROYS_OTHERS_THINGS,
            Feat.CBCL_6_18_Q022_DISOBEDIENT_HOME,
            Feat.CBCL_6_18_Q023_DISOBEDIENT_SCHOOL,
            Feat.CBCL_6_18_Q024_DOESNT_EAT_WELL,
            Feat.CBCL_6_18_Q025_DOESNT_GET_ALONG_OTHERS,
            Feat.CBCL_6_18_Q026_GUILTY_MISBEHAVING,
            Feat.CBCL_6_18_Q027_JEALOUS,
            Feat.CBCL_6_18_Q028_BREAKS_RULES,
            Feat.CBCL_6_18_Q029_FEARS,
            Feat.CBCL_6_18_Q030_FEARS_SCHOOL,
            Feat.CBCL_6_18_Q031_FEARS_BAD,
            Feat.CBCL_6_18_Q032_PERFECT,
            Feat.CBCL_6_18_Q033_FEARS_NO_ONE_LOVES,
            Feat.CBCL_6_18_Q034_OUT_TO_GET,
            Feat.CBCL_6_18_Q035_FEELS_WORTHLESS,
            Feat.CBCL_6_18_Q036_ACCIDENT_PRONE,
            Feat.CBCL_6_18_Q037_FIGHTS,
            Feat.CBCL_6_18_Q038_TEASED,
            Feat.CBCL_6_18_Q039_HANGS_AROUND_TROUBLE,
            Feat.CBCL_6_18_Q040_HEARS_VOICES,
            Feat.CBCL_6_18_Q041_IMPULSIVE,
            Feat.CBCL_6_18_Q042_RATHER_ALONE,
            Feat.CBCL_6_18_Q043_LYING,
            Feat.CBCL_6_18_Q044_BITES_FINGERNAILS,
            Feat.CBCL_6_18_Q045_NERVOUS_TENSE,
            Feat.CBCL_6_18_Q046_TWITCHING,
            Feat.CBCL_6_18_Q047_NIGHTMARES,
            Feat.CBCL_6_18_Q048_NOT_LIKED,
            Feat.CBCL_6_18_Q049_CONSTIPATED,
            Feat.CBCL_6_18_Q050_ANXIOUS,
            Feat.CBCL_6_18_Q051_DIZZY,
            Feat.CBCL_6_18_Q052_FEELS_TOO_GUILTY,
            Feat.CBCL_6_18_Q053_OVEREATING,
            Feat.CBCL_6_18_Q054_OVERTIRED,
            Feat.CBCL_6_18_Q055_OVERWEIGHT,
            Feat.CBCL_6_18_Q056_A_ACHES,
            Feat.CBCL_6_18_Q056_B_HEADACHE,
            Feat.CBCL_6_18_Q056_C_NAUSEA,
            Feat.CBCL_6_18_Q056_D_EYES,
            Feat.CBCL_6_18_Q056_E_RASHES,
            Feat.CBCL_6_18_Q056_F_STOMACHACHES,
            Feat.CBCL_6_18_Q056_G_VOMITING,
            Feat.CBCL_6_18_Q056_H_OTHER,
            Feat.CBCL_6_18_Q057_ATTACKS,
            Feat.CBCL_6_18_Q058_PICKS_SKIN,
            Feat.CBCL_6_18_Q059_SEX_PARTS_PUBLIC,
            Feat.CBCL_6_18_Q060_SEX_PARTS_TOO_MUCH,
            Feat.CBCL_6_18_Q061_POOR_WORK,
            Feat.CBCL_6_18_Q062_CLUMSY,
            Feat.CBCL_6_18_Q063_RATHER_OLDER_KIDS,
            Feat.CBCL_6_18_Q064_RATHER_YOUNGER_KIDS,
            Feat.CBCL_6_18_Q065_REFUSES_TO_TALK,
            Feat.CBCL_6_18_Q066_REPEATS_ACTS,
            Feat.CBCL_6_18_Q067_RUNS_AWAY_HOME,
            Feat.CBCL_6_18_Q068_SCREAMS_A_LOT,
            Feat.CBCL_6_18_Q069_SECRETIVE,
            Feat.CBCL_6_18_Q070_SEES_THINGS,
            Feat.CBCL_6_18_Q071_SELF_CONSCIOUS,
            Feat.CBCL_6_18_Q072_SETS_FIRES,
            Feat.CBCL_6_18_Q073_SEXUAL_PROBLEMS,
            Feat.CBCL_6_18_Q074_CLOWNING,
            Feat.CBCL_6_18_Q075_TOO_SHY,
            Feat.CBCL_6_18_Q076_SLEEPS_LESS,
            Feat.CBCL_6_18_Q077_SLEEPS_MORE,
            Feat.CBCL_6_18_Q078_EASILY_DISTRACTED,
            Feat.CBCL_6_18_Q079_SPEECH_PROBLEM,
            Feat.CBCL_6_18_Q080_STARES_BLANKLY,
            Feat.CBCL_6_18_Q081_STEALS_HOME,
            Feat.CBCL_6_18_Q082_STEALS_OUTSIDE,
            Feat.CBCL_6_18_Q083_STORES_MANY_THINGS,
            Feat.CBCL_6_18_Q084_STRANGE_BEHAVIOR,
            Feat.CBCL_6_18_Q085_STRANGE_IDEAS,
            Feat.CBCL_6_18_Q086_STUBBORN,
            Feat.CBCL_6_18_Q087_CHANGES_MOOD,
            Feat.CBCL_6_18_Q088_SULKS,
            Feat.CBCL_6_18_Q089_SUSPICIOUS,
            Feat.CBCL_6_18_Q090_OBSCENE_LANGUAGE,
            Feat.CBCL_6_18_Q091_TALKS_KILLING_SELF,
            Feat.CBCL_6_18_Q092_TALKS_WALKS_SLEEP,
            Feat.CBCL_6_18_Q093_TALKS_TOO_MUCH,
            Feat.CBCL_6_18_Q094_TEASES_A_LOT,
            Feat.CBCL_6_18_Q095_TANTRUMS,
            Feat.CBCL_6_18_Q096_THINKS_SEX_TOO_MUCH,
            Feat.CBCL_6_18_Q097_THREATENS,
            Feat.CBCL_6_18_Q098_THUMB_SUCKING,
            Feat.CBCL_6_18_Q099_TOBACCO,
            Feat.CBCL_6_18_Q100_TROUBLE_SLEEPING,
            Feat.CBCL_6_18_Q101_SKIPS_SCHOOL,
            Feat.CBCL_6_18_Q102_UNDERACTIVE,
            Feat.CBCL_6_18_Q103_UNHAPPY,
            Feat.CBCL_6_18_Q104_UNUSUALLY_LOUD,
            Feat.CBCL_6_18_Q105_DRUGS,
            Feat.CBCL_6_18_Q106_VANDALISM,
            Feat.CBCL_6_18_Q107_WETS_SELF,
            Feat.CBCL_6_18_Q108_WETS_BED,
            Feat.CBCL_6_18_Q109_WHINING,
            Feat.CBCL_6_18_Q110_WISHES_TO_BE_OPP_SEX,
            Feat.CBCL_6_18_Q111_WITHDRAWN,
            Feat.CBCL_6_18_Q112_WORRIES,
        ],
        [
            Feat.CBCL_6_18_ANXIOUS_DEPRESSED_T_SCORE,
            Feat.CBCL_6_18_WITHDRAWN_DEPRESSED_T_SCORE,
            Feat.CBCL_6_18_SOMATIC_COMPLAINTS_T_SCORE,
            Feat.CBCL_6_18_SOCIAL_PROBLEMS_T_SCORE,
            Feat.CBCL_6_18_THOUGHT_PROBLEMS_T_SCORE,
            Feat.CBCL_6_18_ATTENTION_PROBLEMS_T_SCORE,
            Feat.CBCL_6_18_RULE_BREAKING_BEHAVIOR_T_SCORE,
            Feat.CBCL_6_18_AGGRESSIVE_BEHAVIOR_T_SCORE,
            Feat.CBCL_6_18_INTERNALIZING_PROBLEMS_T_SCORE,
            Feat.CBCL_6_18_EXTERNALIZING_PROBLEMS_T_SCORE,
            Feat.CBCL_6_18_OBSESSIVE_COMPULSIVE_PROBLEMS_T_SCORE,
            Feat.CBCL_6_18_SLUGGISH_COGNITIVE_TEMPO_T_SCORE,
            Feat.CBCL_6_18_STRESS_PROBLEMS_T_SCORE,
            Feat.CBCL_6_18_DSM5_CONDUCT_PROBLEMS_T_SCORE,
            Feat.CBCL_6_18_DSM5_SOMATIC_PROBLEMS_T_SCORE,
            Feat.CBCL_6_18_DSM5_OPPOSITIONAL_DEFIANT_T_SCORE,
            Feat.CBCL_6_18_DSM5_ATTENTION_DEFICIT_HYPERACTIVITY_T_SCORE,
            Feat.CBCL_6_18_DSM5_ANXIETY_PROBLEMS_T_SCORE,
            Feat.CBCL_6_18_DSM5_DEPRESSIVE_PROBLEMS_T_SCORE,
        ],
        Feat.CBCL_6_18_TOTAL_PROBLEMS_T_SCORE,
    )

    CBCL_1_5 = (
        "cbcl_1_5",
        "cbcl_1_5",
        [
            Feat.CBCL_1_5_Q001_ACHES_PAINS,
            Feat.CBCL_1_5_Q002_ACTS_TOO_YOUNG,
            Feat.CBCL_1_5_Q003_AFRAID_NEW,
            Feat.CBCL_1_5_Q004_AVOIDS_EYE,
            Feat.CBCL_1_5_Q005_CONCENTRATE,
            Feat.CBCL_1_5_Q006_RESTLESS_HYPERACTIVE,
            Feat.CBCL_1_5_Q007_OUT_OF_PLACE,
            Feat.CBCL_1_5_Q008_WAITING,
            Feat.CBCL_1_5_Q009_CHEWS_EDIBLE,
            Feat.CBCL_1_5_Q010_CLINGS,
            Feat.CBCL_1_5_Q011_SEEKS_HELP,
            Feat.CBCL_1_5_Q012_CONSTIPATED,
            Feat.CBCL_1_5_Q013_CRIES,
            Feat.CBCL_1_5_Q014_CRUEL_TO_ANIMALS,
            Feat.CBCL_1_5_Q015_DEFIANT,
            Feat.CBCL_1_5_Q016_DEMANDS_MET,
            Feat.CBCL_1_5_Q017_DESTROYS_OWN_THINGS,
            Feat.CBCL_1_5_Q018_DESTROYS_FAMILY_THINGS,
            Feat.CBCL_1_5_Q019_DIARRHEA,
            Feat.CBCL_1_5_Q020_DISOBEDIENT,
            Feat.CBCL_1_5_Q021_CHANGE_ROUTINE,
            Feat.CBCL_1_5_Q022_SLEEP_ALONE,
            Feat.CBCL_1_5_Q023_ANSWER_PEOPLE,
            Feat.CBCL_1_5_Q024_EAT_WELL,
            Feat.CBCL_1_5_Q025_GET_ALONG,
            Feat.CBCL_1_5_Q026_HAVE_FUN,
            Feat.CBCL_1_5_Q027_FEEL_GUILTY,
            Feat.CBCL_1_5_Q028_GO_OUT_HOME,
            Feat.CBCL_1_5_Q029_EASILY_FRUSTRATED,
            Feat.CBCL_1_5_Q030_EASILY_JEALOUS,
            Feat.CBCL_1_5_Q031_EATS_NOT_FOOD,
            Feat.CBCL_1_5_Q032_FEARS_ANIMALS,
            Feat.CBCL_1_5_Q033_FEELINGS_HURT,
            Feat.CBCL_1_5_Q034_ACCIDENT_PRONE,
            Feat.CBCL_1_5_Q035_GETS_FIGHTS,
            Feat.CBCL_1_5_Q036_GETS_INTO_EVERYTHING,
            Feat.CBCL_1_5_Q037_UPSET_SEPARATED,
            Feat.CBCL_1_5_Q038_TROUBLE_SLEEP,
            Feat.CBCL_1_5_Q039_HEADACHES,
            Feat.CBCL_1_5_Q040_HITS_OTHERS,
            Feat.CBCL_1_5_Q041_HOLDS_BREATH,
            Feat.CBCL_1_5_Q042_HURTS_ANIMALS,
            Feat.CBCL_1_5_Q043_UNHAPPY_REASON,
            Feat.CBCL_1_5_Q044_ANGRY_MOODS,
            Feat.CBCL_1_5_Q045_NAUSEA,
            Feat.CBCL_1_5_Q046_TWITCHING,
            Feat.CBCL_1_5_Q047_NERVOUS_TENSE,
            Feat.CBCL_1_5_Q048_NIGHTMARES,
            Feat.CBCL_1_5_Q049_OVEREATING,
            Feat.CBCL_1_5_Q050_OVERTIRED,
            Feat.CBCL_1_5_Q051_PANIC_NO_REASON,
            Feat.CBCL_1_5_Q052_PAINFUL_BOWEL,
            Feat.CBCL_1_5_Q053_ATTACKS_PEOPLE,
            Feat.CBCL_1_5_Q054_PICKS_NOSE,
            Feat.CBCL_1_5_Q055_PLAYS_SEX_PARTS,
            Feat.CBCL_1_5_Q056_CLUMSY,
            Feat.CBCL_1_5_Q057_PROBLEMS_EYES,
            Feat.CBCL_1_5_Q058_PUNISHMENT_BEHAV_CHANGE,
            Feat.CBCL_1_5_Q059_SHIFTS_ACTIVITY,
            Feat.CBCL_1_5_Q060_RASHES,
            Feat.CBCL_1_5_Q061_REFUSES_EAT,
            Feat.CBCL_1_5_Q062_REFUSES_ACTIVE_GAMES,
            Feat.CBCL_1_5_Q063_ROCKS_HEAD,
            Feat.CBCL_1_5_Q064_RESISTS_BED,
            Feat.CBCL_1_5_Q065_RESISTS_TOILET_TRAINING,
            Feat.CBCL_1_5_Q066_SCREAMS,
            Feat.CBCL_1_5_Q067_UNRESPONSIVE_AFFECTION,
            Feat.CBCL_1_5_Q068_SELF_CONSCIOUS,
            Feat.CBCL_1_5_Q069_SELFISH,
            Feat.CBCL_1_5_Q070_LITTLE_AFFECTION,
            Feat.CBCL_1_5_Q071_LITTLE_INTEREST_AROUND,
            Feat.CBCL_1_5_Q072_LITTLE_FEAR_GETTING_HURT,
            Feat.CBCL_1_5_Q073_TOO_SHY,
            Feat.CBCL_1_5_Q074_SLEEPS_LESS,
            Feat.CBCL_1_5_Q075_PLAYS_BOWEL_MOVEMENTS,
            Feat.CBCL_1_5_Q076_SPEECH_PROBLEM,
            Feat.CBCL_1_5_Q077_STARES_SPACE,
            Feat.CBCL_1_5_Q078_STOMACHACHES,
            Feat.CBCL_1_5_Q079_SADNESS_EXCITEMENT,
            Feat.CBCL_1_5_Q080_STRANGE_BEHAVIOR,
            Feat.CBCL_1_5_Q081_STUBBORN,
            Feat.CBCL_1_5_Q082_CHANGES_MOOD,
            Feat.CBCL_1_5_Q083_SULKS,
            Feat.CBCL_1_5_Q084_TALKS_SLEEP,
            Feat.CBCL_1_5_Q085_TEMPER_TANTRUMS,
            Feat.CBCL_1_5_Q086_CONCERNED_NEATNESS,
            Feat.CBCL_1_5_Q087_FEARFUL,
            Feat.CBCL_1_5_Q088_UNCOOPERATIVE,
            Feat.CBCL_1_5_Q089_UNDERACTIVE,
            Feat.CBCL_1_5_Q090_UNHAPPY,
            Feat.CBCL_1_5_Q091_LOUD,
            Feat.CBCL_1_5_Q092_UPSET_NEW_PEOPLE,
            Feat.CBCL_1_5_Q093_VOMITING,
            Feat.CBCL_1_5_Q094_WAKES_UP_NIGHT,
            Feat.CBCL_1_5_Q095_WANDERS_AWAY,
            Feat.CBCL_1_5_Q096_WANTS_ATTENTION,
            Feat.CBCL_1_5_Q097_WHINING,
            Feat.CBCL_1_5_Q098_WITHDRAWN,
            Feat.CBCL_1_5_Q099_WORRIES,
        ],
        [
            Feat.CBCL_1_5_EMOTIONALLY_REACTIVE_T_SCORE,
            Feat.CBCL_1_5_ANXIOUS_DEPRESSED_T_SCORE,
            Feat.CBCL_1_5_SOMATIC_COMPLAINTS_T_SCORE,
            Feat.CBCL_1_5_WITHDRAWN_T_SCORE,
            Feat.CBCL_1_5_SLEEP_PROBLEMS_T_SCORE,
            Feat.CBCL_1_5_ATTENTION_PROBLEMS_T_SCORE,
            Feat.CBCL_1_5_AGGRESSIVE_BEHAVIOR_T_SCORE,
            Feat.CBCL_1_5_INTERNALIZING_PROBLEMS_T_SCORE,
            Feat.CBCL_1_5_EXTERNALIZING_PROBLEMS_T_SCORE,
            Feat.CBCL_1_5_STRESS_PROBLEMS_T_SCORE,
            Feat.CBCL_1_5_DSM5_DEPRESSIVE_PROBLEMS_T_SCORE,
            Feat.CBCL_1_5_DSM5_ANXIETY_PROBLEMS_T_SCORE,
            Feat.CBCL_1_5_DSM5_AUTISM_SPECTRUM_PROBLEMS_T_SCORE,
            Feat.CBCL_1_5_DSM5_ATTENTION_DEFICIT_HYPERACTIVITY_T_SCORE,
            Feat.CBCL_1_5_DSM5_OPPOSITIONAL_DEFIANT_T_SCORE,
        ],
        Feat.CBCL_1_5_TOTAL_PROBLEMS_T_SCORE,
    )

    BMS = ("bms", "basic_medical_screening", None, None, None)
    ACI = ("aci", "approximated_cognitive_impairment", None, None, None)
    ADI = ("adi", "area_deprivation_index", None, None, None)
    ASR = ("asr", "asr", None, None, None)
    BHA = ("bha", "background_history_adult", None, None, None)
    BHC = ("bhc", "background_history_child", None, None, None)
    BHS = ("bhs", "background_history_sibling", None, None, None)
    CLR = ("clr", "clinical_lab_results", None, None, None)
    CDV = ("cdv", "core_descriptive_variables", None, None, None)
    IR = ("ir", "individuals_registration", None, None, None)
    IQ = ("iq", "iq", None, None, None)
    ROLES = ("roles", "roles", None, None, None)
    SRGD = ("srgd", "self_reported_genetic_diagnosis", None, None, None)
    SRS_2_ASR = ("srs_2_asr", "srs-2_adult_self_report", None, None, None)
    SRS_2_DA = ("srs_2_DA", "srs2_dependent_adult", None, None, None)
    SRS_2_SA = ("srs_2_sa", "srs2_school_age", None, None, None)
    V3 = ("v3", "vineland-3", None, None, None)
    ADOS_O_1 = ("ados_o_1", "ados/ados_original_module_1", None, None, None)
    ADOS_O_2 = ("ados_o_2", "ados/ados_original_module_2", None, None, None)
    ADOS_O_3 = ("ados_o_3", "ados/ados_original_module_3", None, None, None)
    ADOS_O_4 = ("ados_o_4", "ados/ados_original_module_4", None, None, None)
    ADOS_2_T = ("ados_2_t", "ados/ados_2_toddler", None, None, None)
    ADOS_2_1 = ("ados_2_1", "ados/ados_2_module_1", None, None, None)
    ADOS_2_2 = ("ados_2_2", "ados/ados_2_module_2", None, None, None)
    ADOS_2_3 = ("ados_2_3", "ados/ados_2_module_3", None, None, None)
    ADOS_2_4 = ("ados_2_4", "ados/ados_2_module_4", None, None, None)
