# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


def mit_lm(xdf_input, xselekt_cols, xtarget, outputs=True, do_norm=True, shuffle_cols=False):
    """
        Erkläre die Beziehung zwischen Zielvariable und ausgewählten numerische Werte anhand eines LM
        Beispiel:
            xselekt_cols=['CareerSatisfaction', 'HoursPerWeek', 'JobSatisfaction', 'StackOverflowSatisfaction']
            mit_lm(xdfInput, xselekt_cols, xtarget = "Salary")
    """
    dfres = xdf_input.copy()
    dfres = dfres.reset_index(drop=True)
    if shuffle_cols:
        np.random.shuffle(xselekt_cols)
    X = dfres[xselekt_cols]
    y = dfres[xtarget]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30)  # random_state=42
    lm_model = LinearRegression(normalize=do_norm)
    lm_model.fit(X_train, y_train)
    xpred = lm_model.predict(X)
    Xvis = X.copy()
    Xvis["pred"] = xpred
    Xvis["IST"] = y
    Xvis["delta"] = [a - b for a, b in Xvis[["IST", "pred"]].values]

    r2_perf = r2_score(y, xpred)
    rmse_perf = np.sqrt(mean_squared_error(y, xpred))
    xcoeffsDF = pd.DataFrame({
        "feature": xselekt_cols,
        "gewicht": lm_model.coef_
    })
    grundwert = lm_model.intercept_
    xcoeffsDF = xcoeffsDF.sort_values(by="gewicht")
    xcoeffsDF = xcoeffsDF.reset_index(drop=True)
    if outputs:
        print("Grundwert:")
        print(grundwert)
        print(xcoeffsDF, "\n")
        print("delta = IST - pred")
        print(Xvis.describe())
        print("R^2 {}".format(r2_perf))
        print("RMSE {}".format(rmse_perf))
        _ = plt.subplots(figsize=(10, 8))
        Xvis.delta.hist(grid=False, bins=50)
        plt.title("delta = IST - Voraussage")
        plt.tight_layout()
        plt.show()
        Xvis.plot.scatter("IST", "pred", s=3)
        plt.xlabel(xtarget)
        plt.ylabel("pred({})".format(xtarget))
        plt.title("{} vs pred({})".format(xtarget, xtarget))
        plt.tight_layout()
        plt.show()

    return Xvis, r2_perf, rmse_perf, xcoeffsDF, grundwert


def kateg_felder_schwellenwert_auswahl_mit_lm(xdfcat_dummy, target, starte_ab=0.8, do_plot=True):
    """ Ergibt eine Liste von ausgewählten Features, die ein LM Modell optimieren
        target: eine Spalte wird in xdf_cat mit pd.merge() eingebunden
        BeispieL:
            target = xdf_num[["Salary"]]
            xLdf = kateg_felder_schwellenwert_auswahl_mit_lm(xdf_cat, target, starte_ab = 0.5)
        Zugriff dann mit:
            xLdf.iloc[k].xselekt_cols
    """
    # Für die kategorialen Variablen: np.where((X.sum() > cutoff) == True)[0] ergibt die Spalten-Indexen
    X = xdfcat_dummy.copy()
    xtarget = target.columns[0]
    xL = []
    dict_coefs = {}
    while starte_ab > 0.0:
        cutoff = starte_ab * X.shape[0]
        xselekt_cols = [X.columns[t] for t in np.where((X.sum() > cutoff) == True)[0]]
        if len(xselekt_cols) == 0:
            starte_ab -= 0.01
            continue
        dflm = pd.merge(X[xselekt_cols], target, left_index=True, right_index=True)
        Xvis, r2_perf, rmse_perf, xcoeffsDF, grundwert = mit_lm(dflm, xselekt_cols, xtarget, outputs=False)
        xL.append([cutoff, r2_perf, len(xselekt_cols), xselekt_cols, grundwert])
        dict_coefs[cutoff] = xcoeffsDF
        starte_ab -= 0.01
    xLdf = pd.DataFrame(xL)
    xLdf.columns = ["cutoff", "r2_perf", "#cols", "xselekt_cols", "grundwert"]
    xLdf = xLdf[xLdf.r2_perf > 0]
    xLdf = xLdf[xLdf.grundwert > 0]
    xLdf = xLdf.sort_values(by="r2_perf", ascending=False)
    xLdf = xLdf.reset_index(drop=True)
    print(xLdf)
    if do_plot:
        xLdf.plot.scatter("r2_perf", "cutoff", s=4)
        plt.tight_layout()
        plt.ylabel("Schwellenwert")
        plt.xlabel("R^2")
        plt.show()
    u = xLdf['r2_perf'] * [10.0 / t for t in xLdf['#cols'].values]
    return xLdf, dict_coefs, u[u == u.max()].index[0]


def simulation_lm(xdf_input, xselekt_cols, xtarget, nsim=300):
    """ Gegeben eine Auswahl an Features,
        führt den LM mehrmals aus und findet die Durchschnittliche Werte für die Parameters des Modells
    """
    if True:
        xL = []
        for simk in range(nsim):
            _ = mit_lm(xdf_input, xselekt_cols, xtarget, outputs=False, do_norm=False, shuffle_cols=True)
            Xvis, r2_perf, rmse_perf, xcoeffsDF, grundwert = _
            xL.append([r2_perf, rmse_perf, xcoeffsDF, grundwert])
        xdict_cols = {}
        grundwerte = []
        for _, _, xr, grundwert in xL:
            grundwerte.append(grundwert)
            for xcol, gewicht in xr.values:
                if xcol not in list(xdict_cols.keys()):
                    xdict_cols[xcol] = [gewicht]
                else:
                    xdict_cols[xcol].append(gewicht)
        xdf_cols_stats = pd.DataFrame(xdict_cols)
        xdf_cols_stats["Grundwert"] = grundwerte
        print(xdf_cols_stats.describe())

    xdf_cols_stats.hist(bins=25, grid=False)
    plt.autoscale(enable=True)
    plt.show()

    print("Durchschnitte: \n", xdf_cols_stats.mean())
    parameter_means = xdf_cols_stats.mean().to_dict()

    return xdf_cols_stats, parameter_means


def do_lm_berechnung(xrecord, xtarget, parameter_means, voraussage=False):
    xformel = []
    for xkey in xrecord.keys():
        if xkey != xtarget:
            xformel.append(
                [xkey, parameter_means[xkey], xrecord[xkey]]
            )
    xformel = pd.DataFrame(xformel)
    xformel.columns = ["feature", "gewicht", "wert"]
    xformel["gewicht * wert"] = [t * v for t, v in xformel[["gewicht", "wert"]].values]
    xformel = xformel.sort_values(by="gewicht")
    xformel = xformel.reset_index(drop=True)
    x = xformel["gewicht * wert"].sum()
    b = parameter_means["Grundwert"]
    if not voraussage:
        return xformel, x, b
    else:
        return x + b


# Beispiel Datesätze
def zeige_lm_berechnung(xdf_input, xselekt_cols, xtarget, parameter_means):
    xrecord = xdf_input[xselekt_cols + [xtarget]].sample(10).iloc[5].to_dict()
    xformel, x, b = do_lm_berechnung(xrecord, xtarget, parameter_means)
    print(xformel)
    print("\n")
    print("\tSUMME       : ", x)
    print("\tGrundwert   : ", b)
    print("\t          =>  ", x + b)
    print("\tTatsächlich : ", xrecord[xtarget])
