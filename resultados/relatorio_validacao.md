# Relatório de validação — agro-credit-risk-model

Dados **sintéticos** (processo gerador em `src/agro_credit/dados.py`). Os números medem o método, não uma carteira real.

Desenvolvimento: safras [2018, 2019, 2020, 2021] · Fora do tempo: safras [2022, 2023]

## Modelo de concessão (só informação disponível antes do plantio)

| Modelo | Amostra | AUC | Gini | KS | Brier | Default obs. | PD média |
|---|---|---|---|---|---|---|---|
| logistica | teste_no_tempo | 0.6659 | 0.3318 | 0.2458 | 0.0417 | 0.045 | 0.0451 |
| logistica | fora_do_tempo | 0.6875 | 0.3751 | 0.2841 | 0.0587 | 0.065 | 0.0455 |
| gradient_boosting | teste_no_tempo | 0.6361 | 0.2722 | 0.242 | 0.0426 | 0.045 | 0.0442 |
| gradient_boosting | fora_do_tempo | 0.6613 | 0.3226 | 0.2533 | 0.0592 | 0.065 | 0.0446 |

PSI do score (desenvolvimento × fora do tempo): logistica 0.0026, gradient_boosting 0.0029

## Estabilidade das variáveis (PSI)

| Variável | PSI |
|---|---|
| area_ha | 0.0029 |
| vol_chuva_hist | 0.0022 |
| alavancagem | 0.0019 |
| anos_relacionamento | 0.0011 |
| ndvi_hist_medio | 0.001 |
| atrasos_12m | 0.0001 |
| tem_seguro | 0.0 |
| zarc_risco | 0.0 |

## Modelo comportamental (acrescenta anomalia de chuva e NDVI da safra)

Fora do tempo: AUC 0.7205, Gini 0.441, KS 0.3285. Serve para monitorar a carteira durante a safra, não para conceder.

## Calibração por decil — logística, fora do tempo

|   decil |   n |   pd_media |   default_observado |
|--------:|----:|-----------:|--------------------:|
|       1 | 600 |     0.011  |              0.025  |
|       2 | 600 |     0.017  |              0.03   |
|       3 | 600 |     0.0216 |              0.04   |
|       4 | 600 |     0.0261 |              0.0333 |
|       5 | 600 |     0.0312 |              0.035  |
|       6 | 600 |     0.0372 |              0.0567 |
|       7 | 600 |     0.0445 |              0.0683 |
|       8 | 600 |     0.055  |              0.08   |
|       9 | 600 |     0.0735 |              0.095  |
|      10 | 600 |     0.1384 |              0.1867 |

## Calibração por safra e UF — logística, fora do tempo

| Safra | UF | n | Default obs. | PD média |
|---|---|---|---|---|
| 2022 | BA | 300 | 0.0633 | 0.0542 |
| 2022 | GO | 427 | 0.0375 | 0.0406 |
| 2022 | MG | 432 | 0.037 | 0.0361 |
| 2022 | MT | 616 | 0.0325 | 0.0441 |
| 2022 | PR | 607 | 0.0708 | 0.0412 |
| 2022 | RS | 618 | 0.1456 | 0.0576 |
| 2023 | BA | 317 | 0.0757 | 0.0548 |
| 2023 | GO | 437 | 0.0458 | 0.0446 |
| 2023 | MG | 460 | 0.0435 | 0.0339 |
| 2023 | MT | 556 | 0.0558 | 0.0426 |
| 2023 | PR | 602 | 0.0498 | 0.0423 |
| 2023 | RS | 628 | 0.0971 | 0.0553 |

## Perda esperada (PD × LGD × EAD), fora do tempo

Modelo: 126,138,876 · Realizada (simulada): 194,423,414. LGD por garantia é premissa ilustrativa; EAD = exposição sintética.

## Coeficientes da logística (variáveis padronizadas)

| variavel                   |   coeficiente |
|:---------------------------|--------------:|
| cat__garantia_aval         |         0.547 |
| num__atrasos_12m           |         0.353 |
| num__ndvi_hist_medio       |        -0.336 |
| num__alavancagem           |         0.325 |
| num__anos_relacionamento   |        -0.275 |
| cat__uf_MG                 |        -0.231 |
| num__area_ha               |        -0.181 |
| num__vol_chuva_hist        |         0.153 |
| cat__garantia_penhor_safra |        -0.152 |
| cat__uf_MT                 |         0.13  |
| num__zarc_risco            |         0.106 |
| cat__uf_PR                 |        -0.103 |
