# agro-credit-risk-model

Modelo de probabilidade de inadimplência (PD) para crédito rural, com variáveis financeiras e agroclimáticas, validação fora do tempo e monitoramento de estabilidade e calibração.

> **Dados sintéticos.** Todo o processo gerador está em `src/agro_credit/dados.py`. Os números medem o método, não uma carteira real.

## O problema

Na concessão de crédito rural, o banco conhece as finanças do produtor, o zoneamento de risco climático (ZARC) e o histórico da lavoura, mas não sabe como será o clima da safra. Um choque climático regional muda a inadimplência sem mudar o perfil dos tomadores. Por isso o monitoramento usual por PSI pode não perceber o problema.

## O que o projeto faz

1. Gera uma carteira de 18 mil operações em seis safras (2018–2023) e seis UFs, com estiagem forte no Sul em 2022–2023.
2. **Modelo de concessão:** usa só informação disponível antes do plantio (alavancagem, atrasos, relacionamento, garantia, seguro, ZARC, NDVI histórico, volatilidade histórica de chuva). A anomalia climática da safra fica fora, para não haver vazamento.
3. **Modelo comportamental:** acrescenta a anomalia de chuva e de NDVI observada durante a safra, para monitorar a carteira antes do vencimento.
4. Compara regressão logística e gradient boosting, com teste no tempo e **validação fora do tempo** (desenvolvimento 2018–2021, validação 2022–2023).
5. Mede AUC, Gini, KS, Brier, PSI do score e das variáveis, calibração por decil e por safra×UF, e perda esperada (PD × LGD × EAD).

## Resultados (seed 42)

| Modelo | Amostra | AUC | Gini | KS |
|---|---|---|---|---|
| Logística — concessão | teste no tempo | 0,666 | 0,332 | 0,246 |
| Logística — concessão | fora do tempo | 0,688 | 0,375 | 0,284 |
| Gradient boosting — concessão | fora do tempo | 0,661 | 0,323 | 0,253 |
| Logística — comportamental | fora do tempo | 0,721 | 0,441 | 0,329 |

**Achado principal:** o PSI do score entre desenvolvimento e validação fica em 0,003, e o de todas as variáveis abaixo de 0,01 (estável). Mesmo assim, a inadimplência fora do tempo sobe de 4,5% para 6,5%, e o modelo de concessão prevê 4,6% em média. No RS em 2022 a inadimplência observada é de 14,6% contra 5,8% prevista. A deriva é climática e não aparece no PSI: o monitoramento precisa de calibração por safra e região e de um modelo comportamental com dado de satélite e chuva.

A logística supera o gradient boosting aqui em parte porque o processo gerador é logístico. Em dado real a comparação precisa ser refeita.

Relatório completo: `resultados/relatorio_validacao.md` · Métricas: `resultados/metricas.json`.

## Como rodar

    uv venv && uv pip install -r requirements.txt
    .venv/bin/python -m pytest -q
    PYTHONPATH=src .venv/bin/python -m agro_credit.executar

## Limitações

- LGD por tipo de garantia é premissa ilustrativa; a EAD é sintética.
- Variáveis agroclimáticas reais (NDVI Sentinel-2, chuva, ZARC por município) podem ser integradas a partir das mesmas colunas.
- Não há tratamento de reject inference nem de safras com renegociação.

## Licença

MIT — ver `LICENSE`.
