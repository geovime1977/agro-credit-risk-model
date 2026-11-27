"""Gerador de carteira sintética de crédito rural.

Os dados são sintéticos: o processo gerador está todo neste arquivo, para que
qualquer resultado possa ser auditado. Nenhum número aqui descreve uma carteira
real.

Separação temporal das informações:
- conhecidas na concessão (antes do plantio): finanças, histórico, ZARC,
  vigor histórico da lavoura (NDVI médio de safras anteriores), volatilidade
  histórica de chuva da região;
- observadas durante a safra: anomalia de NDVI e de chuva da safra corrente.
  Elas causam inadimplência, mas não podem entrar no modelo de concessão.
"""

import numpy as np
import pandas as pd

SAFRAS = list(range(2018, 2024))
UFS = ["RS", "PR", "MT", "GO", "MG", "BA"]
CULTURAS = ["soja", "milho", "cafe"]
GARANTIAS = ["alienacao_fiduciaria", "penhor_safra", "aval"]

# choque climático médio (anomalia de chuva, desvio-padrão) por safra e região;
# 2022 e 2023 trazem estiagem forte no Sul, o que cria deriva fora do tempo
CHOQUE = {(2022, "RS"): -1.6, (2022, "PR"): -1.1, (2023, "RS"): -1.2, (2021, "MT"): -0.5}

VOL_CHUVA_UF = {"RS": 1.2, "PR": 1.0, "MT": 0.7, "GO": 0.8, "MG": 0.9, "BA": 1.3}

VARIAVEIS_CONCESSAO = [
    "uf", "cultura", "garantia", "area_ha", "alavancagem", "atrasos_12m",
    "anos_relacionamento", "tem_seguro", "zarc_risco", "ndvi_hist_medio",
    "vol_chuva_hist",
]
VARIAVEIS_SAFRA = ["anomalia_chuva", "anomalia_ndvi"]


def gerar_base(n_por_safra: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    partes = []
    for safra in SAFRAS:
        n = n_por_safra
        uf = rng.choice(UFS, n, p=[0.2, 0.2, 0.2, 0.15, 0.15, 0.1])
        cultura = rng.choice(CULTURAS, n, p=[0.6, 0.3, 0.1])
        area = np.exp(rng.normal(5.0, 1.0, n))
        alav = np.clip(rng.lognormal(-0.9, 0.55, n), 0.05, 3.0)
        atrasos = rng.poisson(0.35, n)
        relac = rng.integers(0, 25, n)
        garantia = rng.choice(GARANTIAS, n, p=[0.45, 0.35, 0.20])
        seguro = rng.random(n) < 0.55
        zarc = rng.choice([20, 30, 40], n, p=[0.6, 0.3, 0.1])
        ndvi_hist = np.clip(rng.normal(0.70, 0.06, n), 0.4, 0.9)
        vol = np.array([VOL_CHUVA_UF[u] for u in uf]) * rng.uniform(0.85, 1.15, n)

        choque = np.array([CHOQUE.get((safra, u), 0.0) for u in uf])
        anomalia_chuva = choque + rng.normal(0, 1, n) * vol * 0.6
        anomalia_ndvi = 0.04 * anomalia_chuva + rng.normal(0, 0.03, n)
        quebra = np.clip(-anomalia_chuva, 0, None) * (1 + (cultura == "milho") * 0.3)

        protecao = np.where(seguro, 0.45, 1.0)
        logit = (
            -4.1
            + 1.25 * alav
            + 0.55 * atrasos
            - 0.03 * relac
            + 0.022 * (zarc - 20)
            - 4.0 * (ndvi_hist - 0.70)
            + 0.35 * (vol - 0.9)
            + 0.95 * quebra * protecao
            + np.where(garantia == "aval", 0.35, 0.0)
            - 0.10 * (np.log(area) - 5.0)
            + rng.normal(0, 0.35, n)
        )
        default = rng.random(n) < 1 / (1 + np.exp(-logit))

        partes.append(pd.DataFrame({
            "safra": safra, "uf": uf, "cultura": cultura, "garantia": garantia,
            "area_ha": area.round(1), "alavancagem": alav.round(3),
            "atrasos_12m": atrasos, "anos_relacionamento": relac,
            "tem_seguro": seguro.astype(int), "zarc_risco": zarc,
            "ndvi_hist_medio": ndvi_hist.round(3), "vol_chuva_hist": vol.round(3),
            "anomalia_chuva": anomalia_chuva.round(3),
            "anomalia_ndvi": anomalia_ndvi.round(4),
            "exposicao": (area * rng.uniform(3000, 6000, n)).round(2),
            "default": default.astype(int),
        }))
    return pd.concat(partes, ignore_index=True)
