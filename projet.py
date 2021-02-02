# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 10:57:05 2021

@author: GIANG Cécile
"""

######################### IMPORTATION DES DONNEES #########################

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
    num_product = -1
        
    while(len(data) < data_number + 1):
        
        line=file.readline().strip()
        index = line.find(':')
            
        # Cas ligne vide
        if index == -1:
            continue
            
        # Nom de l'attribut de la donnée traitée
        entry_name = line[:index]
            
        if entry_name not in fields: fields.append(entry_name)
            
        if entry_name == fields[0]:
            num_product += 1
            data.append([])
            
        data[num_product].append(line[index + 2:])
    
    # Suppression du dernier élément
    data.pop()
            
    return np.array(data), np.array(fields)

## Création du dataset
data, fields = parse('data/Cell_Phones_and_Accessories.json', data_number=10000)

            