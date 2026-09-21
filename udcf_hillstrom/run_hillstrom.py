"""
Aplica o modelo UDCF (LBCF, Ai et al. 2022, Secao 4.2.1) na base publica Hillstrom
para estimar o CATE (Conditional Average Treatment Effect) de cada tratamento de
e-mail (Mens E-Mail e Womens E-Mail) contra o grupo controle (No E-Mail), usando
"conversion" como outcome.

Implementacao do UDCF escrita do zero (ver udcf.py) a partir da descricao do
artigo. Nao reutiliza o codigo C++/GRF do repositorio dos autores.
"""

import time
import numpy as np
import pandas as pd

from udcf import UDCFForest

HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)

RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(HILLSTROM_URL)
    print("Base Hillstrom original:", df.shape)
    print(df[[
        "recency", "history", "mens", "womens", "zip_code",
        "newbie", "channel", "segment", "conversion",
    ]].head())
    return df


def build_design(df: pd.DataFrame):
    df = df.copy()

    treat_map = {"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 2}
    df["treatment_code"] = df["segment"].map(treat_map)

    feature_cols = ["recency", "history", "mens", "womens", "newbie"]
    X = df[feature_cols].astype(float).copy()

    zip_dummies = pd.get_dummies(df["zip_code"], prefix="zip", dtype=float)
    channel_dummies = pd.get_dummies(df["channel"], prefix="channel", dtype=float)
    X = pd.concat([X, zip_dummies, channel_dummies], axis=1)

    feature_names = list(X.columns)
    X = X.to_numpy(dtype=float)

    K = 2
    T = np.zeros((len(df), K))
    T[df["treatment_code"].to_numpy() == 1, 0] = 1.0  # T1 = Mens E-Mail
    T[df["treatment_code"].to_numpy() == 2, 1] = 1.0  # T2 = Womens E-Mail

    Y = df["conversion"].to_numpy(dtype=float)

    return X, T, Y, feature_names, df


def main():
    df = load_data()
    X, T, Y, feature_names, df = build_design(df)

    n = X.shape[0]
    n_control = int((T.sum(axis=1) == 0).sum())
    n_t1 = int(T[:, 0].sum())
    n_t2 = int(T[:, 1].sum())
    print(f"\nN total: {n} | controle: {n_control} | Mens E-Mail: {n_t1} | Womens E-Mail: {n_t2}")
    print("Taxa de conversao geral:", Y.mean())
    print("Features usadas:", feature_names)

    forest = UDCFForest(
        n_trees=200,
        subsample_frac=0.5,
        honesty_frac=0.5,
        max_depth=5,
        min_leaf_per_arm=30,
        n_candidates=15,
        m_top=5,
        mtry=None,
        random_state=RANDOM_STATE,
    )

    t0 = time.time()
    forest.fit(X, T, Y)
    t1 = time.time()
    print(f"\nFloresta UDCF treinada ({forest.n_trees} arvores) em {t1 - t0:.1f}s")

    cate = forest.predict(X)
    t2 = time.time()
    print(f"CATE estimado para {n} usuarios em {t2 - t1:.1f}s")

    out = df.copy()
    out["cate_mens_email"] = cate[:, 0]
    out["cate_womens_email"] = cate[:, 1]
    out.to_csv("hillstrom_udcf_cate.csv", index=False)

    print("\n=== Resumo do CATE estimado (UDCF) ===")
    for name, col in [("Mens E-Mail", cate[:, 0]), ("Womens E-Mail", cate[:, 1])]:
        print(f"\nTratamento: {name}")
        print(f"  Media (CATE medio / ATE aproximado): {col.mean():.4f}")
        print(f"  Desvio padrao entre usuarios:         {col.std():.4f}")
        print(f"  Minimo / Maximo:                      {col.min():.4f} / {col.max():.4f}")
        qs = np.quantile(col, [0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"  Quantis (10/25/50/75/90%):             {np.round(qs, 4)}")

    naive_t1 = Y[T[:, 0] == 1].mean() - Y[T.sum(axis=1) == 0].mean()
    naive_t2 = Y[T[:, 1] == 1].mean() - Y[T.sum(axis=1) == 0].mean()
    print("\n=== Diferenca simples de medias (ATE naive, para conferencia) ===")
    print(f"  Mens E-Mail vs controle:   {naive_t1:.4f}")
    print(f"  Womens E-Mail vs controle: {naive_t2:.4f}")

    print("\nArquivo salvo: hillstrom_udcf_cate.csv (inclui CATE por usuario)")


if __name__ == "__main__":
    main()
