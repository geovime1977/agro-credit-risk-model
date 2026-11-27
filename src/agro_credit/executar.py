"""Desenvolve, valida fora do tempo e documenta os modelos. Uso: python -m agro_credit.executar"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from agro_credit.dados import VARIAVEIS_CONCESSAO, VARIAVEIS_SAFRA, gerar_base
from agro_credit.metricas import psi, resumo, tabela_decis
from agro_credit.modelo import gradient_boosting, logistica, separar_fora_do_tempo

SAIDA = Path(__file__).resolve().parents[2] / "resultados"
# premissas ilustrativas de LGD por garantia (não calibradas em dado real)
LGD = {"alienacao_fiduciaria": 0.35, "penhor_safra": 0.50, "aval": 0.70}


def _ajustar(pipe, X, y):
    pipe.fit(X, y)
    return pipe


def _coeficientes(pipe) -> pd.DataFrame:
    nomes = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    t = pd.DataFrame({"variavel": nomes, "coeficiente": coef})
    return t.reindex(t["coeficiente"].abs().sort_values(ascending=False).index).round(3)


def main(seed: int = 42) -> dict:
    SAIDA.mkdir(exist_ok=True)
    df = gerar_base(seed=seed)
    dev, oot = separar_fora_do_tempo(df)
    treino, teste = train_test_split(dev, test_size=0.3, stratify=dev["default"], random_state=seed)

    res = {"base": {
        "n_total": len(df), "n_desenvolvimento": len(dev), "n_fora_do_tempo": len(oot),
        "safras_desenvolvimento": sorted(dev["safra"].unique().tolist()),
        "safras_fora_do_tempo": sorted(oot["safra"].unique().tolist()),
        "taxa_default_por_safra": df.groupby("safra")["default"].mean().round(4).to_dict(),
    }, "concessao": {}}

    modelos = {}
    for nome, fabrica in (("logistica", logistica), ("gradient_boosting", gradient_boosting)):
        pipe = _ajustar(fabrica(VARIAVEIS_CONCESSAO), treino[VARIAVEIS_CONCESSAO], treino["default"])
        modelos[nome] = pipe
        p_teste = pipe.predict_proba(teste[VARIAVEIS_CONCESSAO])[:, 1]
        p_oot = pipe.predict_proba(oot[VARIAVEIS_CONCESSAO])[:, 1]
        p_treino = pipe.predict_proba(treino[VARIAVEIS_CONCESSAO])[:, 1]
        res["concessao"][nome] = {
            "teste_no_tempo": resumo(teste["default"], p_teste),
            "fora_do_tempo": resumo(oot["default"], p_oot),
            "psi_score_dev_vs_oot": round(psi(p_treino, p_oot), 4),
        }

    numericas = [v for v in VARIAVEIS_CONCESSAO if v not in ("uf", "cultura", "garantia")]
    res["psi_variaveis"] = {v: round(psi(dev[v], oot[v]), 4) for v in numericas}

    variaveis_comp = VARIAVEIS_CONCESSAO + VARIAVEIS_SAFRA
    comp = _ajustar(logistica(variaveis_comp), treino[variaveis_comp], treino["default"])
    p_comp = comp.predict_proba(oot[variaveis_comp])[:, 1]
    res["comportamental_logistica_fora_do_tempo"] = resumo(oot["default"], p_comp)

    p_oot = modelos["logistica"].predict_proba(oot[VARIAVEIS_CONCESSAO])[:, 1]
    lgd = oot["garantia"].map(LGD).to_numpy()
    ead = oot["exposicao"].to_numpy()
    res["perda_esperada_fora_do_tempo"] = {
        "perda_esperada_modelo": round(float(np.sum(p_oot * lgd * ead)), 2),
        "perda_realizada_simulada": round(float(np.sum(oot["default"].to_numpy() * lgd * ead)), 2),
        "nota": "LGD por garantia é premissa ilustrativa; EAD = exposição sintética.",
    }

    por_uf = []
    for (safra, uf), g in oot.assign(pd_prev=p_oot).groupby(["safra", "uf"]):
        por_uf.append({"safra": int(safra), "uf": uf, "n": len(g),
                       "default_observado": round(g["default"].mean(), 4),
                       "pd_media_prevista": round(g["pd_prev"].mean(), 4)})
    res["calibracao_safra_uf_oot"] = por_uf

    (SAIDA / "metricas.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    _relatorio(res, _coeficientes(modelos["logistica"]), tabela_decis(oot["default"], p_oot))
    return res


def _relatorio(res, coef, decis):
    c = res["concessao"]
    linhas = [
        "# Relatório de validação — agro-credit-risk-model",
        "",
        "Dados **sintéticos** (processo gerador em `src/agro_credit/dados.py`). Os números medem o método, não uma carteira real.",
        "",
        f"Desenvolvimento: safras {res['base']['safras_desenvolvimento']} · Fora do tempo: safras {res['base']['safras_fora_do_tempo']}",
        "",
        "## Modelo de concessão (só informação disponível antes do plantio)",
        "",
        "| Modelo | Amostra | AUC | Gini | KS | Brier | Default obs. | PD média |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for nome, r in c.items():
        for amostra in ("teste_no_tempo", "fora_do_tempo"):
            m = r[amostra]
            linhas.append(f"| {nome} | {amostra} | {m['auc']} | {m['gini']} | {m['ks']} | {m['brier']} | {m['taxa_default_observada']} | {m['pd_media_prevista']} |")
    linhas += ["", "PSI do score (desenvolvimento × fora do tempo): " + ", ".join(f"{k} {v['psi_score_dev_vs_oot']}" for k, v in c.items()), ""]
    linhas += ["## Estabilidade das variáveis (PSI)", "", "| Variável | PSI |", "|---|---|"]
    linhas += [f"| {k} | {v} |" for k, v in sorted(res["psi_variaveis"].items(), key=lambda x: -x[1])]
    m = res["comportamental_logistica_fora_do_tempo"]
    linhas += [
        "", "## Modelo comportamental (acrescenta anomalia de chuva e NDVI da safra)", "",
        f"Fora do tempo: AUC {m['auc']}, Gini {m['gini']}, KS {m['ks']}. Serve para monitorar a carteira durante a safra, não para conceder.",
        "", "## Calibração por decil — logística, fora do tempo", "", decis.to_markdown(),
        "", "## Calibração por safra e UF — logística, fora do tempo", "",
        "| Safra | UF | n | Default obs. | PD média |", "|---|---|---|---|---|",
    ]
    linhas += [f"| {r['safra']} | {r['uf']} | {r['n']} | {r['default_observado']} | {r['pd_media_prevista']} |" for r in res["calibracao_safra_uf_oot"]]
    pe = res["perda_esperada_fora_do_tempo"]
    linhas += [
        "", "## Perda esperada (PD × LGD × EAD), fora do tempo", "",
        f"Modelo: {pe['perda_esperada_modelo']:,.0f} · Realizada (simulada): {pe['perda_realizada_simulada']:,.0f}. {pe['nota']}",
        "", "## Coeficientes da logística (variáveis padronizadas)", "", coef.head(12).to_markdown(index=False),
    ]
    (SAIDA / "relatorio_validacao.md").write_text("\n".join(linhas) + "\n")


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: r[k] for k in ("concessao", "comportamental_logistica_fora_do_tempo")}, indent=2))
