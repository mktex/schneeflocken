# -*- coding: utf-8 -*-

import seaborn as sbn
import numpy as np
import pandas as pd
from functools import reduce
from contextlib import contextmanager

@contextmanager
def SuppressPandasWarning():
    with pd.option_context("mode.chained_assignment", None):
        yield

# Ergibt die Anzahl oder Prozent der Nullen in einer Spalte
xNullen = lambda df, xcol: df[df[xcol].isnull()].shape[0]
xProzentNullen = lambda df, xcol: df[df[xcol].isnull()].shape[0] / df.shape[0]
xNullenInDF = lambda xdf: pd.DataFrame([(xcol, xNullen(xdf, xcol)) for xcol in xdf.columns])
xProzentNullenInDF = lambda xdf: pd.DataFrame([(xcol, xProzentNullen(xdf, xcol)) for xcol in xdf.columns])

# Ergibt die Durchschnittswerte einer kontinuierlicher Variable pro Wert aus einer Kategorialer Variable
xDurchschnittswertProKateg = lambda df, xkateg, xwert: df[[xkateg, xwert]] \
    .groupby(xkateg).mean().sort_values(by=xwert)


def kateg_werte_liste(xdfInput, xcol, sep=None):
    """ Ergibt die Liste der Werte in kategorialer Variable
        Bleibt der sep None, dann sind die Werte nichts anderes als ein set(xL)
    """
    xL = list(map(lambda x: str(x), xdfInput[xcol].values.tolist()))
    xL = list(filter(lambda x: x is not np.nan, xL))
    if sep is not None:
        xL = list(map(lambda x: list(set(x.split(sep))), xL))
        xL = list(reduce(lambda a, b: a + b, xL))
    xL = [str(t).strip() for t in xL]
    return xL


def frequenz_werte(dfInput, xcol="CousinEducation", prozente=False, sep=None):
    """ Anzahl der Datensätze mit xcol == {Wert} ODER Wert in xcol.split(sep)
    """
    df = dfInput.copy()
    xl = kateg_werte_liste(df, xcol, sep=sep)
    xs = pd.DataFrame({xcol: xl})
    xs['id'] = xs.index.values.tolist()
    xres = xs.groupby(xcol, as_index=True).count()
    xres = xres.sort_values(by="id", ascending=False)
    if prozente:
        xres['id'] = [t / dfInput.shape[0] for t in xres['id'].values.tolist()]
    return xres


def num_cat(xdfInput):
    """ Spalte den Datenbestand vertikal in kategoriale und numerische Teile
        Beispiel:
            xdf_num, xdf_cat = dbeschreiben.num_cat(dfres)
    """
    xdf_cat = xdfInput.select_dtypes(include=['object']).copy()
    xdf_num = xdfInput[list(filter(lambda xc: xc not in xdf_cat.columns, xdfInput.columns))]
    print("\nKategoriale Felder: {}".format(xdf_cat.columns))
    print("\nNumerische Felder: {}".format(xdf_num.columns))
    return xdf_num, xdf_cat


def kategwert2col(xc):
    from string import punctuation
    for c in punctuation:
        xc = xc.replace(c, "")
    xc = "".join(list(map(lambda x: x[0].upper() + (x[1:].lower() if len(x) > 1 else ""),
                          filter(lambda y: y != "",
                                 map(lambda z: z.strip(), xc.split(" "))
                                 )
                          )
                      )
                 )
    return xc


def get_dummies(xdfInput, xcol, xkategWerte):
    # TODO: spark einsetzen
    xL = []
    for ik in range(xdfInput.shape[0]):
        xdatensatz_wert = xdfInput[xcol].iloc[ik]
        xL.append(
            [int(str(t) in str(xdatensatz_wert)) for t in xkategWerte]
        )
    xLdf = pd.DataFrame(xL)
    xLdf.columns = xkategWerte
    return xLdf


def kateg2dummy(xdfInput, sep=None):
    """ Umwandelt die Spalten in Dummy-Variablen mit 1-Hot Encoding
        Falls in der kategorialen Felder mehrere Werte gleich eingesetzt wurden,
        dann kann ein Parameter zB sep=";" verwendet werden, um die voneinander zu trennen
        Beispiel:
            xdfcat_dummy = dbeschreiben.kateg2dummy(xdf_cat, sep=";")
    """
    xL = []
    xdf = xdfInput.copy()
    xdf = xdf.reset_index(drop=True)

    def hatsep(xcol):
        if sep is not None:
            for t in xdf[xcol].values:
                if sep in str(t):
                    return True
        return False

    for xcol in xdf.columns:
        explode = False if sep is None else hatsep(xcol)
        if not explode:
            xdfres = pd.get_dummies(xdf[xcol])
        else:
            xkateg_werte = list(set(kateg_werte_liste(xdfInput, xcol, sep=sep)))
            xkateg_werte.sort()
            xdfres = get_dummies(xdf, xcol, xkateg_werte)
        xdfres.columns = [xcol + "_" + kategwert2col(xc) for xc in xdfres.columns]
        xL.append(xdfres)
    xdfres_gesamt = None
    for i in range(len(xL) - 1):
        if xdfres_gesamt is None:
            xdfres_gesamt = pd.merge(xL[0], xL[1], left_index=True, right_index=True)
        else:
            xdfres_gesamt = pd.merge(xdfres_gesamt, xL[i + 1], left_index=True, right_index=True)
    xdfcat_dummy = xdfres_gesamt
    xdict_count = {};
    xLColumns = list(xdfcat_dummy.columns)
    for i in range(len(xLColumns)):
        xcol = xLColumns[i]
        if xcol not in xdict_count.keys():
            xdict_count[xcol] = 0
        else:
            xdict_count[xcol] += 1
            xLColumns[i] = "{}_{}".format(xLColumns[i], xdict_count[xcol])
    xdfcat_dummy.columns = xLColumns
    return xdfcat_dummy
