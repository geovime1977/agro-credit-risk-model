from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

CATEGORICAS = ["uf", "cultura", "garantia"]


def _numericas(variaveis):
    return [v for v in variaveis if v not in CATEGORICAS]


def logistica(variaveis) -> Pipeline:
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAS),
        ("num", StandardScaler(), _numericas(variaveis)),
    ])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(C=1.0, max_iter=2000))])


def gradient_boosting(variaveis) -> Pipeline:
    prep = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAS),
        ("num", "passthrough", _numericas(variaveis)),
    ])
    clf = HistGradientBoostingClassifier(
        categorical_features=list(range(len(CATEGORICAS))),
        max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=0,
    )
    return Pipeline([("prep", prep), ("clf", clf)])


def separar_fora_do_tempo(df, ultima_safra_treino: int = 2021):
    """Desenvolvimento nas safras até `ultima_safra_treino`; validação nas seguintes."""
    return df[df["safra"] <= ultima_safra_treino], df[df["safra"] > ultima_safra_treino]
