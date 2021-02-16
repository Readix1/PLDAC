# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 10:57:05 2021

@author: GIANG Cécile, LENOIR Romain
"""

######################### IMPORTATION DES DONNEES #########################

import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from scipy.signal import find_peaks

import copy


######################### PREPARATION DES DONNEES #########################

def parse(filename, data_number):
    """ Permet de parser un fichier de données Amazon et crée notre dataset.
        Renvoie la liste des échantillons de données par review et la liste 
        des attributs associés à ces échantillons.
        @param filename: str, nom du fichier à parser
        @data_number: int, nombre de reviews à prendre dans le fichier.
                      Prend les data_number premiers reviews
        @return data: list(str), liste des échantillons de données, chaque
                      échantillon correspondant à un avis posté.
        @return fields: list(str), liste des attributs
    """
    # ------- Ouverture du fichier, initialisation de data et fields
    file = open(filename, 'r')
    data, fields = [], []
    
    # ------- Mise à jour de data et fields par parcours de fichier
    
    for i in range(data_number):
        
        # items = dictionnaire associé à la ligne lue line
        line = file.readline()
        items = json.loads(line)
        
        # ------------------------- MAJ de fields -------------------------
        
        # keys: liste des clés du dictionnaire associé à la ligne courante
        keys = list( items.keys() )
        
        # Cas où la ligne courante présente de nouveaux attributs
        if keys != fields:
            
            new_keys = [k for k in keys if k not in fields]
            fields += new_keys
            
            # Mise à jour des données déjà lues avec les nouvelles clés
            for j in range(i):
                data[j] += [None]*len(new_keys)
                
        # -------------------------- MAJ de data --------------------------
        
        data.append([None]*len(fields))
        
        for index_label in range(len(fields)):
            if fields[index_label] in items.keys():
                data[i][index_label] = items[fields[index_label]]
        
    return data, fields

## Création du dataset
#data, fields = parse('data/cellphones_accessories.json', data_number=10000)


###################### DETECTION DE BURST DE REVIEWS ######################

# Détection de bursts de reviews, qui indiquent des périodes de temps pendant
# lesquelles le nombre de reviews postés est anormalement élevé, et où l'on
# peut suspecter la présence de reviews spam

def reviewPerMonth(data, fields):
    """ Renvoie la liste du nombre de reviews posté par mois dans data.
        @param data: list(str), liste des échantillons de données, chaque
                      échantillon correspondant à un avis posté.
        @param fields: list(str), liste des attributs
        @return X: list(int), liste du nombre de reviews par mois
    """
    index_time = fields.index('reviewTime')
    
    # Temps sous la forme 'mois jour, annee'
    times = [d[index_time] for d in data]
    
    # Normalisation sous la forme (annee, mois, jour)
    times_normalized = []
    for t in times:
        mois, jour, annee = t.split()
        times_normalized.append((int(annee), int(mois), int(jour.strip(','))))
    
    # Tri des temps du plus ancien au plus récent
    times_normalized = sorted(times_normalized)
    
    # Plot le nombre de review par mois
    liste_annees = sorted( np.unique( [t[0] for t in times_normalized] ) )
    liste_mois = sorted( np.unique( [t[1] for t in times_normalized] ) )
    liste_jours = sorted( np.unique( [t[2] for t in times_normalized] ) )
    
    # X: liste du nombre de reviews postés par mois
    X = [0] * (len(liste_annees) * len(liste_mois))
    
    for annee, mois, jour in times_normalized:
        X[liste_annees.index(annee)*len(liste_mois) + liste_mois.index(mois)] += 1
    
    return X


# ------- Détection de bursts de reviews par moyenne mobile

def moyenneMobile(X, width):
    """ Calcule la moyenne mobile de la liste X sur une fenêtre de largeur width.
        @param X: list(int), liste du nombre de reviews par intervalle de temps
                  que l'on veut moyenner
        @param width: int, largeur de la fenêtre de temps
        @param mean: list(float), liste des moyennes mobiles de X
    """
    # Initialisation des moyennes mobiles de X
    mean = [0] * len(X)
    
    # inter: nombre de points pris en compte de chaque côté du point courant
    inter = int(width/2)
    
    for i in range(len(X)):
        if i < inter:
            mean[i] = np.mean( X[ : i + inter + 1] )
        else:
            if len(X)-i < inter:
                mean[i] = np.mean( X[ i - inter : ] )
            else:
                mean[i] = np.mean( X[i - inter : i + inter + 1] )
    return mean    
    
def detectionBurst(data, fields, width, display=False):
    """ Fonction permettant de détecter un burst de reviews dans le temps, 
        pour les données data et fields et une fenêtre de temps de largeur
        width.
        Calcule la différence entre le nombre de reviews dans le temps et la 
        moyenne mobile.
        @param data: list(str), liste des échantillons de données, chaque
                      échantillon correspondant à un avis posté.
        @param fields: list(str), liste des attributs
        @param width: int, largeur de la fenêtre de temps
        @param mean: list(float), liste des moyennes mobiles de X
        @return diff: list(float), liste des différences entre nombre de reviews
                      et moyenne mobile
    """
    # X: nombre de reviews par mois, MM: moyenne mobile de X
    X = reviewPerMonth(data, fields)
    MM = moyenneMobile(X, width)
    
    # diff: liste des différences entre nombre de reviews et moyenne mobile
    diff = np.abs( [ X[i] - MM[i] for i in range(len(X)) ] )
    
    # Affichage graphique:
    if display:
        plt.title('Nombre de reviews par mois')
        plt.plot(X)
        plt.plot(MM)
        #plt.plot(diff)
        
    return X, MM, diff


# ------------ Détection de bursts de reviews par estimation par noyau - KDE

def KDE(data, fields, h):
    """ Fonction permettant de détecter un burst de reviews dans le temps, 
        pour les données data et fields et une fenêtre de temps de largeur h,
        en se basant sur l'estimation par noyau (KDE - Kernel Density Estimation)
        @param data: list(str), liste des échantillons de données, chaque
                      échantillon correspondant à un avis posté.
        @param fields: list(str), liste des attributs
        @param h: int, largeur de la fenêtre de temps
        @return diff: list(float), liste des différences entre nombre de reviews
                      et moyenne mobile
    """
    # X: nombre de reviews par mois, 
    # X_ext: extension de X, remis sous format histogramme
    X = reviewPerMonth(data, fields)
    X_ext = []
    
    for i in range(len(X)):
        X_ext += [i]*X[i]
    
    # Lissage par estimation par noyau (KDE - Kernel Density Estimation)
    X_ext = np.array(X_ext).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(X_ext)
    
    plt.title('Lissage par estimation par noyau - KDE')
    X_plot = np.linspace(0, len(X))
    Y_plot = kde.score_samples(X_plot.reshape(-1,1))
    plt.plot(X_plot, Y_plot)
    
    return Y_plot


############ DETECTION DE REVIEWS SPAMS PAR DEVIATION DE NOTATION ############

# Les reviews spams impliquent généralement une projection incorrecte de la
# qualité d'un produit via une note qui dévie beaucoup de l'opinion générale

def sortReviewsByTime(data, fields):
    """ Trie les reviews de l'échantillon de données data par ordre 
        chronologique.
        @param data: list(str), liste des échantillons de données, chaque
                     échantillon correspondant à un avis posté
        @param fields: list(str), liste des attributs
        @return : list(str), data avec reviews triés chronologiquement
        @return indices: list(int), liste des indices des données triés 
                         chrolonogiquement dans l'échantillon data initial
    """
    index_time = fields.index('reviewTime')
    
    # Liste des dates et des indices de ces dates
    dates = [0] * len(data)
    dates_indices = []
    
    # Temps sous la forme 'mois jour, annee'
    for i in range(len(data)):
        t = data[i][index_time]
        mois, jour, annee = t.split()
        dates[i] = (int(annee), int(mois), int(jour.strip(',')))
    
    # Tri des dates par ordre chronologique et récupération des indices des
    # dates triées
    for date, indice in sorted((e, i) for i, e in enumerate(dates)):
        dates_indices.append(indice)
    
    return [ data[i] for i in dates_indices ], dates_indices


def deviationNotes(data, fields, width, bins, seuil):
    """ Calcule la déviation des notes dans le temps et crée l'histogramme
        correspondant aux déviations entre notes données par les utilisateurs
        note moyenne (moyenne mobile).
        @param data: list(str), liste des échantillons de données, chaque
                     échantillon correspondant à un avis posté
        @param fields: list(str), liste des attributs
        @param width: int, largeur de la fenêtre pour la moyenne mobile
        @param bins: int, nombre d'intervalles dans [0,5] que l'on veut 
                     considérer pour tracer l'histogramme
        @return : float array, effectif des différences par intervalle
    """
    # Tri des reviews de data par ordre chronologique
    data_sorted, indices_sorted = sortReviewsByTime(data, fields)
    
    # Notes des reviews triés chronologiquement, calcul des notes moyennes
    notes = np.array( [ r[0] for r in data_sorted ] )
    notes_moyennes = np.array( moyenneMobile(notes, width) )
    
    # Calcul de la déviation entre notes données et notes moyennes (moyenne mobile)
    deviation_notes = np.abs(notes - notes_moyennes)
    
    # Affichage par histogramme, récupération des effectifs par intervalle
    plt.title('Histogramme de la déviation des notes, pour bins = %d' %bins)
    hist = plt.hist(deviation_notes, np.linspace(0, 5, bins+1))
    
    # retourner les dates des reviews dont la déviation est de plus de 3,25
    indices_suspect = np.where(deviation_notes >= seuil)[0]
    ####### COMMENT DETERMINER LES NOTES A DISCRIMINER #######

    return [indices_sorted[i] for i in indices_suspect]



def showRDperMonth(data, fields, width, bins, seuil):
    """ Renvoie la liste du nombre de reviews posté par mois dans data.
        @param data: list(str), liste des échantillons de données, chaque
                      échantillon correspondant à un avis posté.
        @param fields: list(str), liste des attributs
        @return X: list(int), liste du nombre de reviews par mois
    """
    # ----------------- TRACAGE DU NOMBRE DE REVIEWS PAR MOIS
    
    index_time = fields.index('reviewTime')
    
    # Temps sous la forme 'mois jour, annee'
    times = [d[index_time] for d in data]
    
    # Normalisation sous la forme (annee, mois, jour)
    times_normalized = []
    for t in times:
        mois, jour, annee = t.split()
        times_normalized.append((int(annee), int(mois), int(jour.strip(','))))
    
    # Tri des temps du plus ancien au plus récent
    times_normalized = sorted(times_normalized)
    
    # Plot le nombre de review par mois
    liste_annees = sorted( np.unique( [t[0] for t in times_normalized] ) )
    liste_mois = sorted( np.unique( [t[1] for t in times_normalized] ) )
    liste_jours = sorted( np.unique( [t[2] for t in times_normalized] ) )
    
    # X: liste du nombre de reviews postés par mois
    X = [0] * (len(liste_annees) * len(liste_mois))
    
    for annee, mois, jour in times_normalized:
        X[liste_annees.index(annee)*len(liste_mois) + liste_mois.index(mois)] += 1
    
    
    # ----------------- AFFICHAGE DES DATES OU L'ON SUSPECTE UN SPAM
    
    index_dev = deviationNotes(data, fields, width, bins, seuil)
    data_dev = [data[i] for i in index_dev]
    
    # X: liste des dates où l'on suspecte un review spam
    Y = [0] * (len(liste_annees) * len(liste_mois))
    
    # Temps sous la forme 'mois jour, annee'
    times = [d[index_time] for d in data_dev]
    
    # Normalisation sous la forme (annee, mois, jour)
    times_normalized = []
    for t in times:
        mois, jour, annee = t.split()
        times_normalized.append((int(annee), int(mois), int(jour.strip(','))))
    
    # Tri des temps du plus ancien au plus récent
    times_normalized = sorted(times_normalized)
    
    for annee, mois, jour in times_normalized:
        Y[liste_annees.index(annee)*len(liste_mois) + liste_mois.index(mois)] += 1
    
    # ------- Moyenne mobile
    mm = moyenneMobile(X, 5)
    
    # ------- AFFICHAGE
    plt.figure()
    plt.plot(X)
    plt.plot(mm)
    for e in Y:
        if e != 0:
            plt.scatter(Y.index(e), 0, s=1)
    
    return X, Y


def docSansDoublon(data, fields, nomFichier):
    """
        List * List * str -> List
        data : liste de review (liste)
        fields : liste des attributs de data
        nomFichier: path ou est stocké la liste sans doublon
        
        renvoie data sans les doublons (auteur, texte) et la stock dans nomFichier
    
    """
    # Compteur de doublons
    cpt = 0
    
    if fields.index('reviewerName'):
        iAuteur=fields.index('reviewerName')
        if fields.index('reviewText'):
            iText = fields.index('reviewText')
        else:
            return 0
    else:
        return 0
    
    
    hashs=set()
    file= open(nomFichier, 'w', encoding='utf-8')
    data2=[]   
    
    for i in data:
        if i[iText]:
            h = hash(i[iAuteur]+i[iText])
        else:
            continue
        
        if h not in hashs:
            hashs.add(h)
            json.dump(i, file)
            data2.append(i)
        else:
            cpt += 1
    
    # On affiche le nombre de doublons détectés
    print('\n', str(cpt) , 'doublons non-spams (idReviewer - review) détectés')
    
    return data2


def doublons(data, fields):
    """
        List * List * str -> List
        data : liste de review (liste)
        fields : liste des attributs de data
        nomFichier: path ou est stocké la liste sans doublon
        
        renvoie data sans les doublons (auteur, texte) et la stock dans nomFichier
    
    """
    if fields.index('reviewText'):
        iText = fields.index('reviewText')
        if fields.index("asin"):
            iId = fields.index("asin")
        else:
            return 0
    else:
        return 0
    
    
    hashs=dict()
    res=set()  
    
    for i in range(len(data)):
        if data[i][iText]:
            h = hash(data[i][iText])
        else:
            continue
        
        if h not in hashs:
            hashs[h] =i
        else:
            res.add(hashs[h])
            res.add(i)
           
    return res


def detectFromBurstRD(data, fields, width, bins, seuil, window=4, height=50, distance=10, display=False):
    """ Renvoie la liste des indices des datas de reviews suspects:
        * reviews dont la note dévie beaucoup de la moyenne mobile
        * et qui en plus se trouve près d'un pic (burst de reviews)
        @width
        @param width: int, largeur de la fenêtre pour la moyenne mobile
        @param bins: int, nombre d'intervalles dans [0,5] que l'on veut 
                     considérer pour tracer l'histogramme
        @param seuil: float, valeur à partir de laquelle une déviation est suspecte
        @param window: int, fenetre de temps que l'on considère suspect autour 
                       d'un peak du nombre de reviews
        @param height: int, seuil au dessus duquel on prend nos pics
        @param distance: int, distance min entre 2 pics
        
        @return suspams_ids: list(int), liste des indices des reviews suspectées
                             spam (dans fenêtre de burst + déviation note)
    """
    # Première liste de reviews suspectes: reviews dans un burst temporel
    X, Y = showRDperMonth(data, fields, width, bins, seuil)
    X = np.array(X)
    Y = np.array(Y)
    
    # Trouver la liste les maxima locaux se trouvant au dessus d'un certain seuil height,
    # et à une certaine distance d'autres peaks
    peaks, _ = find_peaks(X, height = height, distance = distance)
    
    # Dans peaks on a maintenant la liste indices des pics (maxima locaux) du 
    # nombre de reviews par mois. On considère les reviews dont les indices se
    # trouvent dans une fenêtre de temps window autour de ces pics
    
    # Liste des indices de pics et leur voisinage direct
    window_ids = []
    w = window // 2
    
    for p in peaks:
        window_ids += [p + i for i in range(-w , w + 1)]
    
    Y = np.where(Y != 0)[0]
    
    # windows_ids: liste des indices de reviews dans une fenêtre temporelle d'un pic
    # Y: liste des indices de reviews dont la note est suspecte (déviation forte)
    # On veut les indices communs des deux listes
    suspams = [i for i in Y if i in window_ids]
    
    # Affichage
    if display:
        plt.figure()
        plt.title('Pics à prendre en compte')
        
        # Affichage des pics
        plt.plot(X)
        plt.plot(peaks, X[peaks], "X")
        
        # Affichage des reviews suspectées comme spam
        for r in suspams:
            plt.scatter(r, 0, s=1)
            
        plt.show()
    
    
    # suspams_ids: Liste des indices des reviews suspectes dans data
    
    # Temps sous la forme 'mois jour, annee'
    index_time = fields.index('reviewTime')
    times = [d[index_time] for d in data]
    
    # Normalisation sous la forme (annee, mois, jour)
    tn = []
    for t in times:
        mois, jour, annee = t.split()
        tn.append((int(annee), int(mois), int(jour.strip(','))))
    
    liste_annees = sorted( np.unique( [t[0] for t in tn] ) )
    liste_mois = sorted( np.unique( [t[1] for t in tn] ) )
    liste_jours = sorted( np.unique( [t[2] for t in tn] ) )
    
    suspams_ids = [i for i in range(len(data)) if liste_annees.index(tn[i][0])*len(liste_mois) + liste_mois.index(tn[i][1]) in suspams]
    
    return set(suspams_ids)


def initScores(data, fields, suspams_ids):
    """ Initialisation des scores utilisateurs, produits et reviews.
        @param suspams_ids = list(int), liste des indices des reviews suspectées
                             spam (dans fenêtre de burst + déviation note)
    """
    # Initilisation des scores utilisateurs à +1
    index_utils = fields.index('reviewerID')
    unique_utils = np.unique( np.array( [r[index_utils] for r in data] ) )
    score_utils = { id_util : 1 for id_util in unique_utils }
    
    # Initilisation des scores produits à +1
    index_prods = fields.index('asin')
    unique_prods = np.unique( np.array( [r[index_prods] for r in data] ) )
    score_prods = { id_prod : 1 for id_prod in unique_prods }

    # Initilisation des scores reviews
    index_revs = fields.index('reviewText')
    suspams_ids = suspams_ids.union(doublons(data, fields))
    score_utils = { id_rev : -1 if id_rev in suspams_ids else 1 for id_rev in range(len(data)) }