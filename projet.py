# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 10:57:05 2021

@author: GIANG Cécile, LENOIR Romain
"""

######################### IMPORTATION DES DONNEES #########################

import json
import matplotlib.pyplot as plt
import numpy as np


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


def moyenneMobile(X, width):
    """ Calcule la moyenne mobile de la liste X sur une fenêtre de temps de
        largeur width.
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
        plt.plot(diff)
        
    return X, MM, diff
