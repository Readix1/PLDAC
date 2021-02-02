# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 10:57:05 2021

@author: GIANG Cécile
"""

######################### IMPORTATION DES DONNEES #########################

import numpy as np
import json
import matplotlib.pyplot as plt
import math


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
    
def detectionBurst(data, fields, fenetre):
    x = reviewPerMonth(data, fields)
    m = moyenneM(fenetre, x)
    dif = np.abs([x[i]-m[i] for i in range(len(x))])
    plt.plot(x)
    plt.plot(m)
    plt.plot(dif)
    return x, m, dif

def moyenneM(fenetre, x ):
    m = [0] * len(x)
    inter=int(fenetre/2)
    for i in range(len(m)):
        
        if i<inter:
            m[i]= np.mean(x[:i+inter+1])
        else:
            if len(x)-i< inter:
                m[i]= np.mean(x[i-inter:])
            else:   
                m[i]=np.mean(x[i-inter:i+inter+1])
    
    return m
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


