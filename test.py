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

def count(filename):
    file = open(filename, 'r')
    count = {}
    
    line = file.readline()
    items = json.loads(line)
    
    while line:
        count[items['asin']] = count.get(items['asin'], 0) + 1
    
    return count
    
def parse_some(to_read, to_save, nprods = 500):
    """ Permet de parser un fichier de données Amazon et crée notre dataset.
        Contrairement à la fonction précédente, lit l'intégralité du fichier 
        et ne sauvegarde que les revues associées aux nprods premiers produits.
        Renvoie la liste des échantillons de données par review et la liste 
        des attributs associés à ces échantillons.
        @param to_read: str, nom du fichier à parser
        @param to_save: str, nom du fichier dans lequel stocker les données
        @nprods: int, nombre de produits à prendre dans le fichier.
        @return data: list(str), liste des échantillons de données, chaque
                      échantillon correspondant à un avis posté.
        @return fields: list(str), liste des attributs
    """
    # ------- Ouverture du fichier, initialisation de data et fields
    fread = open(to_read, 'r')
    fsave = open(to_save, 'w')
    data, fields = [], []
    
    # ------- On maintient la liste des produits rencontrés
    list_prods = set()
    
    # ------- Mise à jour de data et fields par parcours de fichier
    line = fread.readline()
    items = json.loads(line)
    
    i = 0
    
    while line:
        # On rajoute la donnée dans le dataset si on n'a pas atteint nprods produits
        # ou si c'est l'un des produits à garder
        if len(list_prods) < nprods or items['asin'] in list_prods:
            
            # ------------------------- MAJ de fields -------------------------
            
            list_prods.add( items['asin'] )
            fsave.write(line)
            
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
            
            i += 1
        
        line = fread.readline()
        if line:
            items = json.loads(line)
    
    # On remplace les textes de reviews vides par ''
    index_revs = fields.index('reviewText')
    
    for i in range(len(data)):
        if data[i][index_revs]==None:
            data[i][index_revs] = ''
    
    fread.close()
    fsave.close()
        
    return data, fields


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


def doublonsNonSpams(data, fields, nomFichier = 'doublons_non_spams.txt'):
    """
        List * List * str -> List
        data : liste de review (liste)
        fields : liste des attributs de data
        nomFichier: path ou est stocké la liste sans doublon
        
        renvoie data sans les doublons non-spams (auteur, texte, produit) et la stock dans nomFichier
    
    """
    # Compteur de doublons
    cpt = 0
    
    if fields.index('reviewerName'):
        iAuteur=fields.index('reviewerName')
        if fields.index('reviewText'):
            iText = fields.index('reviewText')
            if fields.index('asin'):
                iProd = fields.index('asin')
            else:
                return 0
        else:
            return 0
    else:
        return 0
    
    
    hashs=set()
    file= open(nomFichier, 'w', encoding='utf-8')
    data2=[]   
    
    for i in data:
        if i[iText]:
            try:
                h = hash(i[iAuteur]+i[iText]+i[iProd])
            except:
                h = hash(i[iText]+i[iProd])
        else:
            continue
        
        if h not in hashs:
            hashs.add(h)
            json.dump(i, file)
            data2.append(i)
        else:
            cpt += 1
    
    # On affiche le nombre de doublons détectés
    print('\n', str(cpt) , 'doublons non-spams (idReviewer - review - produit) détectés')
    file.close()
    return data2


def doublonsSpams(data, fields):
    """
        List * List * str -> List
        data : liste de review (liste)
        fields : liste des attributs de data
        nomFichier: path ou est stocké la liste sans doublon
        
        renvoie data sans les doublons (auteur, texte) et la stock dans nomFichier
    
    """
    # Compteur de doublons
    cpt = 0
    
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
            hashs[h] = i
        else:
            if len(data[i][6]) > 20:
                res.add(hashs[h])
                res.add(i)
                cpt += 1
    
    # On affiche le nombre de doublons détectés
    print('\n', str(cpt) , 'doublons spams (<> idReviewer, <> idProduit, review) détectés')
           
    return res


def detectFromBurstRD(data, fields, width, bins, seuil, window=4, height=50, distance=10, display=False):
    """ Renvoie la liste des indices des datas de reviews suspects:
        * reviews dont la note dévie beaucoup de la moyenne mobile
        * et qui en plus se trouve près d'un pic (burst de reviews)
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


class ReviewGraph:
    """ Classe pour la propagation des scores utilisateur - score - produit
        par un algorithme de type PageRank.
    """
    def __init__(self, data, fields, width=6, bins=20, seuil=3.25, window=1, height=50, distance=10):
        """ Constructeur de la classe ReviewGraph.
        """
        self.data = data
        self.fields = fields
        
        # ID des reviews qui sont à fois dans un burst et dont la note dévie beaucoup de la moyenne
        self.suspams_ids = detectFromBurstRD(data, fields, width=width, bins=bins, seuil=seuil, window=window, height=height, distance=distance, display=False)
        
        # ID des reviews doublons potentiellement spams
        self.doublons_ids = doublonsSpams(data, fields)
        
        # Initialisation des scores utilisateur, produits et reviews
        
        # T: Reviewer's trustiness (score utilisateur): initialisation à +1
        self.index_utils = fields.index('reviewerID')
        unique_utils = np.unique( np.array( [r[self.index_utils] for r in data] ) )
        self.score_utils = { id_util : 1 for id_util in unique_utils }
        
        # R: Product reliability (score produit): initialisation à +1
        self.index_prods = fields.index('asin')
        unique_prods = np.unique( np.array( [r[self.index_prods] for r in data] ) )
        self.score_prods = { id_prod : 1 for id_prod in unique_prods }
        
        # H: Review honesty (score review): initialisation selon le score d'entente
        #self.suspams_ids = self.suspams_ids.union(doublons(data, fields))
        #self.score_revs = { id_rev : -1 if id_rev in self.suspams_ids else 1 for id_rev in range(len(data)) }
        self.score_revs = { id_rev : 0 for id_rev in range( len( data ) ) }
        
        # Liste des auteurs de revues en doublons ou dans un burst + déviation
        for ids in self.suspams_ids.union(self.doublons_ids):
            self.score_utils[data[ids][self.index_utils]] = -0.02
            #self.score_prods[data[ids][self.index_prods]] = 0
            
        # Création du dictionnaire des revues par mois: idRevsDate
        self.index_time = fields.index('reviewTime')
        self.idRevsDate = {}
        
        for i in range(len(data)):
            mois, jour, annee = data[i][self.index_time].split()
            if (int(annee), int(mois)) not in self.idRevsDate:
                self.idRevsDate[ (int(annee), int(mois)) ] = []
            self.idRevsDate[ (int(annee), int(mois)) ].append(i)
        
        # Clés de idRevsDate dans l'ordre chronologique
        self.idRevsDate_keys = sorted( self.idRevsDate.keys() )
        
        # Indice de la note dans une revue
        self.index_ratings = fields.index('overall')
        
        # Pour nous faciliter la tâche, on redéfinit la base de données en 
        # indexant par utilisateur ou par produit
        self.db_utils = { id_util : [] for id_util in self.score_utils }
        self.db_prods = { id_prod : [] for id_prod in self.score_prods }
        
        for i in range(len(data)):
            self.db_utils[ data[i][self.index_utils] ].append( (i, data[i]) )
            self.db_prods[ data[i][self.index_prods] ].append( (i, data[i]) )
        
        # Calcul des voisins temporels de chaque revue
        self.neighs = { id_rev : self.getNeighRevs(id_rev, window = window) for id_rev in self.score_revs }
        
        # Initialisation du score d'entente
        self.score_entente = self.initScoreRevs()
        
        # Nombre d'itérations
        self.niter = 0
        
    def getNeighRevs(self, index_rev, window = 4):
        """ Pour trouver les revues publiées dans une fenêtre de temps window
            autour de la revue d'indice i dans data (fenêtre en mois: si elle
            vaut 6 par exemple, on prend les revues publiées entre les 3 mois 
            précédant la revue actuelle et les 3 mois suivant sa publication).
            @param index_rev: int, indice du review dans data
            @param window: int, fenêtre de temps (en mois) considérée
            @return neighRevs: list(int), liste des indices des voisins de 
                               index_rev dans la fenêtre de temps window.
        """
        w = window // 2
        
        # On retrouve l'indice de la revue dans self.idRevsDate
        mois, jour, annee = self.data[index_rev][self.index_time].split()
        i = self.idRevsDate_keys.index( (int(annee), int(mois)) )
        
        # Définition des indices bornant la fenêtre de temps
        imin = i - w if i - w > 0 else 0
        imax = i + w + 1 if i + w + 1 < len( self.idRevsDate_keys ) else len( self.idRevsDate_keys )
        
        # neighRevs: liste des revues publiées dans la fenêtre de temps
        neighRevs = []
        for i in self.idRevsDate_keys[ imin : imax ]:
            neighRevs += self.idRevsDate[i]
        
        return neighRevs
    
    def T(self, id_util):
        """ Reviewer's trustiness (confiance utilisateur).
        """
        # Calcul de la somme des scores honnêteté des revues de l'utilisateur
        h = np.sum( [ self.score_revs[id_rev] for id_rev, rev in self.db_utils[id_util] ] )
        return ( 2 / ( 1 + np.exp(-h) ) ) - 1
        
    def A(self, id_rev, delta = 1):
        """ Agreement score (score d'entente).
            @param delta: float, borne sur la différence entre la note actuelle 
                          et les notes des voisins.
        """
        # On récupère les indices des voisins (dans le sens temporel) de id_rev
        neighs_temp = self.neighs[id_rev]
        
        # Produit correspondant à la revue courante
        id_prod = self.data[id_rev][self.index_prods]
        
        # Parmi les revues voisines, on ne garde que celles qui traitent du même produit
        neighs = []
        
        for id_neigh in neighs_temp:
            if self.data[id_neigh][self.index_prods] == id_prod:
                neighs.append(id_neigh)
                
        # Liste des voisins de notes similaires ou différentes à id_rev
        index_rating = self.fields.index('overall')
        rating = self.data[id_rev][index_rating]
        
        # Compteurs score de confiance des voisins similaires - différents
        cpt_sim = 0
        cpt_dif = 0
        
        for i in neighs:
            if np.abs( rating - self.data[i][index_rating] ) <= delta:
                cpt_sim += self.score_utils[self.data[i][self.index_utils]]
            else:
                cpt_dif += self.score_utils[self.data[i][self.index_utils]]
        
        #print('cpt_sim = ', cpt_sim, '  :  cpt_dif = ', cpt_dif )
        # Calcul et normalisation du score d'entente
        a = cpt_sim - cpt_dif
        a = ( 2 / ( 1 + np.exp(-a) ) ) - 1
        #print(a)
        return a
    
    def H(self, id_rev, delta = 1):
        """ Review honesty (honnêteté revue).
            @param delta: float, borne sur la différence entre la note actuelle 
                          et les notes des voisins.
        """
        return np.abs( self.score_prods[ self.data[id_rev][self.index_prods] ] ) * self.score_entente[id_rev]
                      
    def R(self, id_prod, median = 3):
        """ Product reliability (fiabilité produit).
        """        
        theta = 0
        for i, d in self.db_prods[id_prod]:
            if self.score_utils[ d[self.index_utils] ] > 0:
                theta += self.score_utils[ d[self.index_utils] ] * ( d[self.index_ratings] - median )
        
        return ( 2 / ( 1 + np.exp(-theta) ) ) - 1
    
    def initScoreRevs(self):
        """ Initialisation des scores revues.
            Valeurs:
                * -1 si à la fois doublon, dans un burst et note déviante.
                * -0.75 si doublon ou note déviante
                * sinon: selon le score d'entente
        """
        score_entente = {}
        suspams = self.suspams_ids.intersection(self.doublons_ids)
        print('%d doublons spams dans bursts détectés' % len(suspams))
        
        for id_rev in range( len( self.data ) ):
            if id_rev in suspams:
                score_entente[id_rev] = -1
            else:
                score_entente[id_rev] = self.A(id_rev, delta = 1)
        
        return score_entente
    
    def computeScores(self, delta = 1, median = 3, niter = 50):
        """ Algorithme de type PageRank pour propager les scores.
            @param window: int, fenêtre de temps prise pour prendre le 
                           voisinage d'une revue
            @param delta: float, borne sur la différence entre la note actuelle 
                          et les notes des voisins.
            @param median: float, valeur médiane des notes
            @param niter: int, nombre d'itérations
        """
        self.niter += niter
        
        for iter in range(niter):
            
            # Propagation scores reviews
            for id_rev in self.score_revs:
                self.score_revs[id_rev] = self.H(id_rev, delta = delta)
            
            # Propagation des scores utilisateurs
            for id_util in self.score_utils:
                self.score_utils[id_util] = self.T(id_util)
            
            # Propagation des scores produits
            for id_prod in self.score_prods:
                self.score_prods[id_prod] = self.R(id_prod, median = median)
        
        # On remet les scores des reviews sur 100
        self.score_revs_norms = { id_rev : np.round( (self.score_revs[id_rev] + 1) * 50 ) for id_rev in self.score_revs }

        return self.score_revs_norms 
    
    def display_scores(self):
        """ Affichage des histogrammes des scores.
        """
        plt.figure()
        plt.title('Scores utilisateurs, pour niter = %d' % self.niter)
        plt.xlim(-1,1.1)
        plt.hist( list( self.score_utils.values() ) )
        
        plt.figure()
        plt.title('Scores produits, pour niter = %d' % self.niter)
        plt.xlim(-1,1.1)
        plt.hist( list( self.score_prods.values() ) )
        
        plt.figure()
        plt.title('Scores reviews, pour niter = %d' % self.niter)
        plt.xlim(-1,1.1)
        plt.hist( list( self.score_revs.values() ) )
    
    def detect_susprods(self, seuil=-0.75):
        """ Retourne la liste des produits suspects, ie dont le score est 
            inférieur au seuil.
        """
        return [ k for k , v in self.score_prods.items() if v < seuil ]
    
    def detect_susrevs(self, seuil=-0.75):
        """ Retourne la liste des produits suspects, ie dont le score est 
            inférieur au seuil.
        """
        return [ k for k , v in self.score_revs.items() if v < seuil ]
    
    def get_k_worst(self, k):
        """ Retourne la liste des k produits de plus mauvais scores.
        """
        # Tri dans l'ordre croissant des scores
        scores_sorted = { k : v for k, v in sorted(self.score_prods.items(), key=lambda item: item[1]) }
        return list( scores_sorted.keys() )[:k]
        
    def get_k_best(self, k):
        """ Retourne la liste des k produits de meilleurs scores.
        """
        # Tri dans l'ordre décroissant des scores
        scores_sorted = { k : v for k, v in sorted(self.score_prods.items(), key=lambda item: item[1], reverse=True) }
        return list( scores_sorted.keys() )[:k]
    
    def prod_timeline(self, id_prod):
        """ Trace l'évolution des notes des notes d'un produit dans le temps.
        """
        # On isole les données concernant le produit
        data_prod = [ d for d in self.data if d[self.index_prods] == id_prod ]
        
        # On trie dans l'ordre chronologique
        data_prod = sortReviewsByTime(data_prod, self.fields)[0]
        
        index_time = self.fields.index('reviewTime')
        
        # Temps sous la forme 'mois jour, annee'
        times = [d[index_time] for d in data_prod]
        
        # Normalisation sous la forme (annee, mois, jour)
        times_normalized = []
        for t in times:
            mois, jour, annee = t.split()
            times_normalized.append((int(annee), int(mois), int(jour.strip(','))))
        
        # Plot le nombre de review par mois
        liste_annees = [ t[0] for t in times_normalized ]

        liste_annees = [ i for i in range(min(liste_annees), max(liste_annees) + 1) ]
        liste_mois = [ i for i in range(1,13) ]
        
        # X: moyenne des notes par mois
        X = [[]] * (len(liste_annees) * len(liste_mois))
        
        for i in range(len(times_normalized)):
            annee, mois, jour = times_normalized[i]
            index = liste_annees.index(annee)*len(liste_mois) + liste_mois.index(mois)
            X[index] = X[index] + [data_prod[i][0]]
        
        # On moyenne sur les 10 derniers produits
        X_flattened = [ note for notes_mois in X for note in notes_mois ]
        
        X_mean = copy.deepcopy(X)
        k = 0
        for i in range(len(X)):
            if X[i] != []:
                for j in range(len(X[i])):
                    X_mean[i][j] = np.mean( X_flattened[ max(0,k-5) : min( k+5 , len(X_flattened)) ])
                    k += 1
        
        # On moyenne pour chaque mois
        X = [ np.mean(X[i]) if X[i] != [] else 0 for i in range(len(X)) ]
        X_mean = [ np.mean(X_mean[i]) if X_mean[i] != [] else 0 for i in range(len(X_mean)) ]
        
        # Revues de produit suspectes
        susrevs = self.detect_susrevs(seuil=-0.75)
        
        data_prod_sus = []
        for rev in susrevs:
            if self.data[rev][self.index_prods]==id_prod:
                data_prod_sus.append(rev)
        
        # Temps sous la forme 'mois jour, annee'
        times = [self.data[d][index_time] for d in data_prod_sus]
        
        # Normalisation sous la forme (annee, mois, jour)
        times_normalized = []
        for t in times:
            mois, jour, annee = t.split()
            times_normalized.append((int(annee), int(mois), int(jour.strip(','))))
        
        # Y: indice des revues suspectes
        Y = [0] * (len(liste_annees) * len(liste_mois))
        
        for i in range(len(times_normalized)):
            annee, mois, jour = times_normalized[i]
            index = liste_annees.index(annee)*len(liste_mois) + liste_mois.index(mois)
            Y[index] += 1     
        
        # ------- Moyenne mobile
        mm = moyenneMobile(X, 4)
        
        # ------- AFFICHAGE
        plt.figure()
        plt.plot(X, color='khaki')
        #plt.plot(X_mean, color='red')
        plt.plot(mm, color='gold')
        for e in Y:
            if e != 0:
                plt.scatter(Y.index(e), 0, s=1)
        
    def getScoresRevs(self):
        return self.score_revs
    
    def getScoresUtils(self):
        return self.score_utils
    
    def getScoresProds(self):
        return self.score_prods
    
    def getScoresRevsNorms(self):
        return self.score_revs_norms
    


#data, fields = parse('data/cellphones_accessories.json', data_number = 13000)
#rg = ReviewGraph(data, fields, window = 1)
#rg.computeScores(niter=10)
#scores=rg.getScoresRevsNorms()
#index_utils = fields.index('reviewerID')
#index_prods = fields.index('asin')
#index_revs = fields.index('reviewText')
#index_spams = [ id for id, score in scores.items() if score < 25 ]
#data_spams = { i : data[i] for i in index_spams }
#utils_spams = [ d[index_utils] for k, d in data_spams.items() ]
#prods_spams = [ d[index_prods] for k, d in data_spams.items() ]
#notes = list( rg.getScoresRevsNorms().values() )
#plt.hist(notes)

#score_prods = rg.getScoresProds()
#plt.hist( list( score_prods.values() ) )

#rg.display_scores()
#worst = rg.get_k_worst(5)
#best = rg.get_k_best(5)

###################### DETECTION UTILISATEURS SUSPECTS

"""
# Dictionnaire du nombre de revues spams par utilisateur spam
count = {} 
for i in utils_spams: 
    count[i] = count.get(i, 0) + 1
"""

""" # Afficher les utilisateurs ayant écrit plus de 3 revues spams
for k, v in count.items():
    if v > 2:
        print( k )
"""

""" # l la liste des revues spams d'un auteur
l=[]
for i,d in data_spams.items():
    if d[index_utils]== 'A5JLAU2ARJ0BO':
        l.append(d)
"""

""" A5JLAU2ARJ0BO -> exemple très intéressant
A2426K5QH04Q3Y
A27QXQQOLAMRRR
AWE3YLK1XV1GZ
A206YBBT28XGPJ
"""

""" Retrouver toutes les revues des utilisateurs suspects
cpt=0
for d in data:
    if d[index_utils]=='A5JLAU2ARJ0BO':
        print(d)
        print('\n')
        cpt += 1
print(cpt)
"""

###################### DETECTION PRODUITS SUSPECTS

"""
# Dictionnaire du nombre de revues spams par produit spam
count = {} 
for i in prods_spams: 
    count[i] = count.get(i, 0) + 1
"""

""" # Afficher les utilisateurs ayant écrit plus de 3 revues spams
for k, v in count.items():
    if v > 2:
        print( k )
"""

""" # l la liste des revues spams d'un produit
l=[]
for i,d in data_spams.items():
    if d[index_prods]== 'B0002SYC5O':
        l.append(d)
"""

""" B0002SYC5O
"""


""" 
count_ps = {}
for i in prods_spams: 
    count_ps[i] = count_ps.get(i, 0) + 1
    
    
    
data_bis_spams = [d for d in data_bis if d in data_spams.values()]
data_bis=[d for d in data if d[index_prods]=='B0002SYC5O']

# Note moyenne de B0002SYC5O
np.mean([ d[index_ratings] for d in data if d[index_prods]=='B0002SYC5O' ])  """




""" 



np.where(np.array(list(score_prods.values()))<-0.2)

>> (array([101, 127, 139], dtype=int64),)


list(score_prods.keys())[101]
Out[273]: 'B00008O39I'



data_bis = [ d for d in data if d[index_prods]=='B00008O39I']





A FAIRE: ISOLER LES PRODUITSTSTSTSTSTSTSTSTSTT





"""


"""
count = {} 
for d in data: 
    count[d[index_prods]] = count.get(d[index_prods], 0) + 1

prods = [ k for k, v in count.items() if v >= 100]
data_bis = [d for d in data if d[index_prods] in prods]



"""

""" EXPLICATION DU PROBLEME

D'où vient notre problème ?
- cpt_sim et cpt_dif très élevés, même avec un delta à 2
- on essaie de réduire la fenêtre de temps ?


--> réduire la fenêtre de temps: window = 1
--> augmenter delta
--> vérifier que neighs ne contient bien que des reveurs sur les mêmes produits
--> changer le lambda de la fonction sigmoide ? fonction tanh ?
--> normaliser la différence (cpt_sim-ct_diff) et on change la pente de la fonction
    sigmoide ?

X = np.linspace(-1000,1000,1000)
Y = [ (2 / (1+np.exp(-0.01*x)))-1 for x in X ]
#Y = [np.tanh(x) for x in X]
plt.figure()
plt.title('Fonction (2 / (1 - exp(-x))) - 1')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(X,Y,color='lavender')

"""

""" 
Trouver des produits suspects

susprods = rg.detect_susprods(seuil=-0.95)
count=[0]*len(susprods)
for i in range(len(susprods)):
    prod=susprods[i]
    for d in data:
        if d[index_prods]==prod:
            count[i] += 1
print(count)

id_prod = susprods[np.argmax(count)]


"""

"""  Pour afficher les scores finaux des susmaps_ids, suspams (histogramme)

score_revs = rg.getScoresRevs()
suspams_ids = detectFromBurstRD(data, fields, width=6, bins=20, seuil=3.25, window=1, height=50, distance=10, display=False)
doublons_ids = doublonsSpams(data, fields)
suspams = suspams_ids.intersection(doublons_ids)

score_suspams_ids = { i : score_revs[i] for i in suspams_ids }
score_suspams = { i : score_revs[i] for i in suspams }

sum( list( score_suspams_ids.values() ) )
"""