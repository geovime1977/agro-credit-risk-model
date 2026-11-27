import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss


def ks(y, score) -> float:
    """Máxima distância entre as distribuições acumuladas de score de bons e maus."""
    y = np.asarray(y)
    score = np.asarray(score)
    ordem = np.argsort(-score)
    y_ord = y[ordem]
    maus = np.cumsum(y_ord) / max(y_ord.sum(), 1)
    bons = np.cumsum(1 - y_ord) / max((1 - y_ord).sum(), 1)
    return float(np.max(np.abs(maus - bons)))


def gini(y, score) -> float:
    return float(2 * roc_auc_score(y, score) - 1)


def psi(esperado, observado, bins: int = 10) -> float:
    """Population Stability Index com cortes nos quantis da base de referência."""
    esperado = np.asarray(esperado, dtype=float)
    observado = np.asarray(observado, dtype=float)
    cortes = np.unique(np.quantile(esperado, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    e = np.histogram(esperado, cortes)[0] / len(esperado)
    o = np.histogram(observado, cortes)[0] / len(observado)
    e = np.clip(e, 1e-6, None)
    o = np.clip(o, 1e-6, None)
    return float(np.sum((o - e) * np.log(o / e)))


def resumo(y, pd_prevista) -> dict:
    return {
        "auc": round(float(roc_auc_score(y, pd_prevista)), 4),
        "gini": round(gini(y, pd_prevista), 4),
        "ks": round(ks(y, pd_prevista), 4),
        "brier": round(float(brier_score_loss(y, pd_prevista)), 4),
        "taxa_default_observada": round(float(np.mean(y)), 4),
        "pd_media_prevista": round(float(np.mean(pd_prevista)), 4),
    }


def tabela_decis(y, pd_prevista) -> pd.DataFrame:
    """Calibração por decil de PD prevista (decil 10 = maior risco)."""
    df = pd.DataFrame({"y": y, "pd": pd_prevista})
    df["decil"] = pd.qcut(df["pd"].rank(method="first"), 10, labels=range(1, 11))
    t = df.groupby("decil", observed=True).agg(
        n=("y", "size"), pd_media=("pd", "mean"), default_observado=("y", "mean")
    )
    return t.round(4)
