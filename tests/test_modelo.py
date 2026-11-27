import numpy as np
import pytest

from agro_credit.dados import VARIAVEIS_CONCESSAO, VARIAVEIS_SAFRA, gerar_base
from agro_credit.metricas import ks, psi
from agro_credit.modelo import separar_fora_do_tempo


def test_ks_separacao_perfeita_e_aleatoria():
    y = np.array([0] * 50 + [1] * 50)
    assert ks(y, np.arange(100)) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    assert ks(rng.integers(0, 2, 5000), rng.random(5000)) < 0.06


def test_psi_mesma_distribuicao_e_deslocada():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 20000)
    assert psi(a, rng.normal(0, 1, 20000)) < 0.01
    assert psi(a, rng.normal(0.5, 1, 20000)) > 0.1


def test_gerador_reprodutivel_e_taxa_plausivel():
    a, b = gerar_base(500, seed=7), gerar_base(500, seed=7)
    assert a.equals(b)
    assert 0.03 < a["default"].mean() < 0.20


def test_concessao_nao_usa_informacao_da_safra():
    assert not set(VARIAVEIS_SAFRA) & set(VARIAVEIS_CONCESSAO)


def test_separacao_fora_do_tempo_sem_sobreposicao():
    dev, oot = separar_fora_do_tempo(gerar_base(200))
    assert dev["safra"].max() < oot["safra"].min()
