"""Sklearn Transformer to build base features for scoring."""

from numpy import nan
from pandas import DataFrame, notna, to_datetime
from sklearn.base import BaseEstimator, TransformerMixin

naf_expert_2 = {
    "01_expert": [
        "0111Z",
        "0112Z",
        "0113Z",
        "0114Z",
        "0116Z",
        "0119Z",
        "0121Z",
        "0122Z",
        "0123Z",
        "0124Z",
        "0125Z",
        "0126Z",
        "0127Z",
        "0128Z",
        "0129Z",
        "0130Z",
        "0141Z",
        "0142Z",
        "0143Z",
        "0144Z",
        "0145Z",
        "0146Z",
        "0147Z",
        "0149Z",
        "0150Z",
        "0161Z",
        "0162Z",
        "0163Z",
        "0164Z",
        "0170Z",
    ],
    "02_expert": ["0210Z", "0220Z", "0230Z", "0240Z"],
    "03_expert": ["0311Z", "0312Z", "0321Z", "0322Z"],
    "17_expert": [
        "1711Z",
        "1712Z",
        "1721A",
        "1721B",
        "1721C",
        "1722Z",
        "1723Z",
        "1724Z",
        "1729Z",
    ],
    "20_expert": [
        "2011Z",
        "2012Z",
        "2013A",
        "2013B",
        "2014Z",
        "2015Z",
        "2016Z",
    ],
    "21_expert": ["2110Z", "2120Z"],
    "22_expert": [
        "2211Z",
        "2219Z",
        "2221Z",
        "2222Z",
        "2223Z",
        "2229A",
        "2229B",
    ],
    "26_expert": [
        "2611Z",
        "2612Z",
        "2620Z",
        "2630Z",
        "2640Z",
        "2651A",
        "2651B",
        "2652Z",
        "2660Z",
        "2670Z",
        "2680Z",
    ],
    "27_expert": [
        "2711Z",
        "2712Z",
        "2720Z",
        "2731Z",
        "2732Z",
        "2733Z",
        "2740Z",
        "2751Z",
        "2752Z",
    ],
    "28_expert": [
        "2811Z",
        "2812Z",
        "2813Z",
        "2814Z",
        "2815Z",
        "2821Z",
        "2822Z",
        "2823Z",
        "2824Z",
        "2825Z",
        "2829A",
        "2829B",
        "2830Z",
        "2841Z",
        "2849Z",
    ],
    "35_expert": [
        "3511Z",
        "3512Z",
        "3513Z",
        "3214Z",
        "3521Z",
        "3522Z",
        "3523Z",
        "3530Z",
    ],
    "36_expert": ["3600Z"],
    "37_expert": ["3700Z"],
    "38_expert": ["3811Z", "3812Z", "3821Z", "3822Z", "3831Z", "3832Z"],
    "39_expert": ["3900Z"],
    "46_expert": [
        "4611Z",
        "4612A",
        "4612B",
        "4613Z",
        "4614Z",
        "4615Z",
        "4616Z",
        "4617A",
        "4617B",
        "4618Z",
        "4619A",
        "4619B",
        "4621Z",
        "4622Z",
        "4623Z",
        "4624Z",
        "4631Z",
        "4632A",
        "4632B",
        "4632C",
        "4633Z",
        "4634Z",
        "4635Z",
        "4636Z",
        "4637Z",
        "4638A",
        "4638B",
        "4639A",
        "4639B",
        "4651Z",
        "4652Z",
        "4690Z",
    ],
    "52_expert": [
        "5210A",
        "5210B",
        "5221Z",
        "5222Z",
        "5223Z",
        "5224A",
        "5224B",
        "5229A",
        "5229B",
    ],
    "53_expert": ["5310Z", "5320Z"],
    "64_expert": ["6411Z", "6419Z", "6430Z", "6491Z", "6492Z", "6499Z"],
    "65_expert": ["6511Z", "6512Z", "6520Z", "6530Z"],
    "69_expert": ["6910Z", "6920Z"],
    "84_expert": [
        "8411Z",
        "8412Z",
        "8413Z",
        "8421Z",
        "8422Z",
        "8423Z",
        "8424Z",
        "8425Z",
        "8430A",
        "8430B",
        "8430C",
    ],
    "85_expert": [
        "8510Z",
        "8520Z",
        "8531Z",
        "8532Z",
        "8541Z",
        "8542Z",
        "8551Z",
        "8552Z",
        "8553Z",
        "8559A",
        "8559B",
        "8560Z",
    ],
    "86_expert": [
        "8610Z",
        "8621Z",
        "8622A",
        "8622B",
        "8622C",
        "8623Z",
        "8690A",
        "8690B",
        "8690C",
        "8690D",
        "8690E",
        "8690F",
    ],
    "87_expert": [
        "8710A",
        "8710B",
        "8710C",
        "8720A",
        "8720B",
        "8730A",
        "8730B",
        "8790A",
        "8790B",
    ],
}


naf_expert_2 = {v: k for k, v_list in naf_expert_2.items() for v in v_list}


def get_naf_expert_2(naf: str):
    """récupération du code naf expert de niveau 2"""

    return naf_expert_2.get(naf, naf)


def build_ratio(X: DataFrame, numerators: str, denominator: str):
    """Builds the ratio of numerators over denominator"""

    # filtre des feature qui ne sont pas des str
    numerators = [c for c in numerators if X[c].dtype != "object"]

    # division des numerateurs
    x_divided = X[numerators].divide(
        X[denominator].replace(0, nan), axis="index"
    )

    # changement du nom des colonnes
    X[[n + "_" + denominator for n in numerators]] = x_divided

    return X


def attribute_luc_irpro(X: DataFrame):
    """Attribution de la LUC/IRPRO et decoupage."""

    # attribution IRPRO ou LUC
    note_bal = nan
    if (X["CODECANALDISTRIBUTPARTENAIRE"] == "BANQ") & (
        X["CODETYPEMODELE"] == "GLES"
    ):
        note_bal = X["NOTE_LUC"]
    elif X["CODECANALAPPORTPARTENAIRE"] == "LCL":
        note_bal = X["NOTE_IRPRO"]

    X["NOTEBAL"] = note_bal

    # découpage note IRPRO ou LUC
    note_bal_bup = nan
    if note_bal in ["A", "B", "C", "D", "E", "U", "1", "2"]:
        note_bal_bup = "R1"
    elif note_bal in ["F", "G", "T", "Y", "3", "4", "5"]:
        note_bal_bup = "R2"
    elif note_bal in ["H", "6", "7"]:
        note_bal_bup = "R3"
    elif note_bal in ["I", "8"]:
        note_bal_bup = "R4"
    elif note_bal in ["J", "9"]:
        note_bal_bup = "R5"
    elif note_bal in ["K", "W", "V", "10", "11"]:
        note_bal_bup = "R6"

    X["NOTEBALBUP"] = note_bal_bup

    return X


def top_note_uti(X: DataFrame) -> DataFrame:
    """Computes TOP_NOTE_UTI: which grade to use between NOR, NOFC and SRR"""
    X["TOP_NOR"] = 0
    if X["NOTE_NOR"] in [
        "A",
        "A+",
        "B",
        "B+",
        "C",
        "C+",
        "C-",
        "D",
        "D+",
        "D-",
        "E",
        "E+",
        "E-",
        "F",
        "Z",
    ]:
        if notna(X["ANCIENNETENOR"]) and (X["ANCIENNETENOR"] <= 450):
            X["TOP_NOR"] = 1

    X["TOP_NOFC"] = 0
    if X["NOTE_NOFC"] not in ["", "NR", "ND"]:
        if notna(X["ANCIENNETENOFC"]) and (X["ANCIENNETENOFC"] <= 450):
            X["TOP_NOFC"] = 1

    X["TOP_SRR"] = 0
    if X["NOTE_SRR"] != "XX":
        if notna(X["ANCIENNETESRR"]) and (X["ANCIENNETESRR"] < 31):
            X["TOP_SRR"] = 1

    if X["CODESEGMENTBALOISCTP"] not in ["02", "13"]:
        X["SEGMENT"] = "CORP"
    else:
        X["SEGMENT"] = "RETAIL"

    X["NOTE_UTI"] = "NA"
    if (X["TOP_SRR"] == 1) and (X["SEGMENT"] == "RETAIL"):
        X["NOTE_UTI"] = "SRR"
    elif (X["TOP_NOR"] == 1) and (X["SEGMENT"] == "CORP"):
        X["NOTE_UTI"] = "NOR"
    elif (X["TOP_NOFC"] == 1) and (X["SEGMENT"] == "CORP"):
        X["NOTE_UTI"] = "NOFC"

    return X


def attribute_nor_nofc(X: DataFrame):
    """Attribution de la note NOR ou NOFC"""

    # attribution de la note
    nor_nofc = nan
    if X["TOP_NOR"] == 1:
        nor_nofc = X["NOTE_NOR"]

    elif X["TOP_NOFC"] == 1:
        nor_nofc = X["NOTE_NOFC"]

    X["NOR_NOFC"] = nor_nofc

    return X


def attriute_srr_nor_nofc(X: DataFrame):
    """Attribution de la note SRR ou NOR ou NOFC"""

    # attribution de la note
    note_risque = "NA"
    if X["NOTE_UTI"] == "SRR":
        note_risque = X["NOTE_SRR"]
    elif X["NOTE_UTI"] == "NOR":
        note_risque = X["NOTE_NOR"]
    elif X["NOTE_UTI"] == "NOFC":
        note_risque = X["NOTE_NOFC"]

    X["NOTE_RISQUE"] = note_risque

    return X


def transcodification_bdf(X: DataFrame):
    """Transcodification de la note Banque de France post MAJ 2022."""

    # transcodification
    if notna(X["DATE_ARRETE_BDF"]):
        if X["DATE_ARRETE_BDF"] >= "2022-01-08":
            note_bdf_transco = nan
            if X["COTECREDIT"] == "0":
                note_bdf_transco = "0"
            if X["COTECREDIT"] == "1+":
                note_bdf_transco = "3++"
            if X["COTECREDIT"] == "1":
                note_bdf_transco = "3+"
            if X["COTECREDIT"] == "1-":
                note_bdf_transco = "3"
            if X["COTECREDIT"] == "2+":
                note_bdf_transco = "4+"
            if X["COTECREDIT"] == "2":
                note_bdf_transco = "4+"
            if X["COTECREDIT"] == "2-":
                note_bdf_transco = "4+"

            if X["COTECREDIT"] == "3+":
                note_bdf_transco = "4"
            if X["COTECREDIT"] == "3":
                note_bdf_transco = "4"
            if X["COTECREDIT"] == "3-":
                note_bdf_transco = "4"
            if X["COTECREDIT"] == "4+":
                note_bdf_transco = "4"
            if X["COTECREDIT"] == "4":
                note_bdf_transco = "5+"
            if X["COTECREDIT"] == "4-":
                note_bdf_transco = "5+"

            if X["COTECREDIT"] == "5+":
                note_bdf_transco = "5+"
            if X["COTECREDIT"] == "5":
                note_bdf_transco = "5"
            if X["COTECREDIT"] == "5-":
                note_bdf_transco = "5"
            if X["COTECREDIT"] == "6+":
                note_bdf_transco = "5"
            if X["COTECREDIT"] == "6":
                note_bdf_transco = "6"
            if X["COTECREDIT"] == "6-":
                note_bdf_transco = "6"
            if X["COTECREDIT"] == "7":
                note_bdf_transco = "7"
            if X["COTECREDIT"] == "8":
                note_bdf_transco = "8"
            if X["COTECREDIT"] == "9":
                note_bdf_transco = "9"
            if X["COTECREDIT"] == "P":
                note_bdf_transco = "P"

            X["COTECREDIT"] = note_bdf_transco

    return X


class dataCleaner(BaseEstimator, TransformerMixin):
    """Buils base features used for model allocation and scoring"""

    def __init__(self, n_jobs=-1, min_freq=0.04):
        """Calcul des variables du modeles et des composantes de l algorithme,
        ajout de variables expertes.

        """

        self.features = None
        self.n_jobs = n_jobs
        self.min_freq = min_freq

    def fit(self, X, y=None):
        """Stores the dataframe's features"""
        self.features = list(X.columns)

        return self

    def transform(self, X, y=None):
        """Computes base features"""
        # construction cotecredit
        X["COTECREDIT"] = X["NOTEBDFCTP"].apply(
            lambda u: u[1:] if notna(u) else u
        )

        # construction des différences de dates
        dates = [
            "DATESITUATIONRISQPIRECTP",
            "DATEDERNIERREFUSCTP",
            "DATEDERNIERACCORDCTP",
            "DATECLOTUREBILAN",
            "DATEIMMATCTP",
            "DATE_ARRETE_NOR",
            "DATE_ARRETE_NOFC",
            "DATE_ARRETE_SRR",
        ]
        date_demande = to_datetime(X["DATEDEMANDE"])
        for date in dates:
            delta = "".join(["DIF", date])
            X[delta] = (
                date_demande.dt.year - to_datetime(X[date]).dt.year
            ) * 12 + (date_demande.dt.month - to_datetime(X[date]).dt.month)
        X = X.rename(
            {
                "DIFDATECLOTUREBILAN": "ANCIENNETELIASSE",
                "DIFDATEIMMATCTP": "ANCIENNETECONTREPARTIE",
                "DIFDATE_ARRETE_NOR": "ANCIENNETENOR",
                "DIFDATE_ARRETE_NOFC": "ANCIENNETENOFC",
                "DIFDATE_ARRETE_SRR": "ANCIENNETESRR",
            },
            axis=1,
        )

        # différenciation entre les X0 et les autres 0
        X.loc[X["NOTEBDFCTP"] == "X0", "COTECREDIT"] = "X0"
        X["PAYDEX"] = X["PAYDEX"].astype(float)

        # correction valeurs négatives impossibles
        mask_outlier = X["EXPOPOTENTIELLEBRUTEGROUPE"] < 0
        X.loc[mask_outlier, "EXPOPOTENTIELLEBRUTEGROUPE"] = 0
        mask_outlier = X["ANCIENNETECONTREPARTIE"] < 0
        X.loc[mask_outlier, "ANCIENNETECONTREPARTIE"] = nan

        # Recréation de la variable ENDETTEMENTTOTAL
        X["ENDETTEMENTTOTAL"] = X["DETTESINF1AN"] + X["EXIGIBLEATERME"]

        # Suppression des liasses fiscales de plus de 18 mois
        # Toutes les variables de la table liassefiscaleetratios ###
        var_liasse = [
            "CHIFFREAFFAIRES",
            "EXCEDENTBRUTEXPLOIT",
            "TOTALACTIF",
            "ENDETTEMENTTOTAL",
            "FONDSPROPRES",
        ]

        # suppression des données de liasses fiscales de plus de 18 de mois
        # l'ancienneté de la liasse est en jour
        old_liasses = X["ANCIENNETELIASSE"] > 18
        X.loc[old_liasses, var_liasse] = nan

        # suppression des liasses de mauvaises qualités
        bad_liasse = X["INDICATEURETATLIASSE"].isin(["0", "1", "2", "3"])
        X.loc[bad_liasse, var_liasse] = nan

        # correction de MTENCOURSNOTE5CONTREPARTIE
        mtencours = [
            "MTENCOURSNOTE5CONTREPARTIE",
            "MTENCOURSNOTE4CONTREPARTIE",
            "MTENCOURSNOTE3CONTREPARTIE",
            "MTENCOURSNOTE2CONTREPARTIE",
            "MTENCOURSNOTE1CONTREPARTIE",
        ]
        X[mtencours] = X[mtencours].where(X[mtencours] >= 0, 0)

        # VARIABLE RATIO #
        numerators = [
            "ENDETTEMENTTOTAL",
            "EXCEDENTBRUTEXPLOIT",
            "ANCIENNETECONTREPARTIE",
        ]
        denominators = ["TOTALACTIF", "MTDEMANDE", "CHIFFREAFFAIRES"]

        # calcul du ratio entre les numerateurs et denominateurs
        for denominator in denominators:
            X = build_ratio(X, numerators, denominator)

        # top grades
        X = X.apply(top_note_uti, axis=1)

        # VARIABLES BANQUES #
        # correctif DIFDATESITUATIONRISQPIRECTP
        mask_bq = X["CODECANALDISTRIBUTPARTENAIRE"] == "BANQ"
        cond = X["DIFDATESITUATIONRISQPIRECTP"] < 0
        X.loc[cond & mask_bq, "DIFDATESITUATIONRISQPIRECTP"] = 0
        cond = X["SITUATIONRISQPIRECTP"] == "SAIN"
        X.loc[cond & mask_bq, "DIFDATESITUATIONRISQPIRECTP"] = nan

        # Note baloise BUP IRPRO et GL LUC
        X = X.apply(attribute_luc_irpro, axis=1)
        X = X.apply(attribute_nor_nofc, axis=1)
        X.loc[~mask_bq, "NOR_NOFC"] = nan

        # VARIABLE EXPERTES #
        # construction du naf expert
        X["CODENAFCTP_EXPERT_2"] = X["CODENAFCTP_5"].apply(get_naf_expert_2)

        # calcul de la note de risque utilisee pour le scoring
        X = X.apply(attriute_srr_nor_nofc, axis=1)

        # transcodification de la note BDF
        X = X.apply(transcodification_bdf, axis=1)

        # calcul de la premiere date de demande
        date_role = to_datetime(X["DATE_CREATION_ROLE"])
        date_role_na = to_datetime(X["DATE_CREATION_ROLE_NA"])
        first_date_demande = date_role.where(~date_role.isna(), date_role_na)
        first_date_demande = first_date_demande.apply(lambda u: str(u)[:10])
        X["FIRST_DATE_DEMANDE"] = first_date_demande.replace("NaT", nan)

        # calcul de l anciennete de la premiere date de demande
        d_anciennete = date_demande - to_datetime(X["FIRST_DATE_DEMANDE"])
        d_anciennete = d_anciennete.apply(lambda u: u.days / 30.5).fillna(0)
        X["ANCIENNETE_CLIENT"] = d_anciennete
        X.loc[X["ANCIENNETE_CLIENT"] < 0, "ANCIENNETE_CLIENT"] = 0

        # Montant en note 5 sur montant encours
        X["MTENCOURS"] = X[mtencours].sum(axis=1)

        # montant encours apres demande par rapport a CAF ou FP
        X["MTENCOURSFUTUR"] = X["MTENCOURS"] + X["MTDEMANDE"]
        mte_fp = X["MTENCOURSFUTUR"] / X["FONDSPROPRES"].replace(0, nan)
        X["MTENCOURSFUTUR_FONDSPROPRES"] = mte_fp

        # suppression de valeurs abberentes
        # X['DIFDATESITUATIONRISQPIRECTP'].quantile(0.999) = 178
        mask_q_999 = X["DIFDATESITUATIONRISQPIRECTP"] > 178
        X.loc[mask_q_999, "DIFDATESITUATIONRISQPIRECTP"] = nan

        return X
