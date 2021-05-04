# -*- coding: utf-8 -*-
"""
Created on Tue May  4 09:40:31 2021

@author: GIANG
"""

""" --- ETAPE 1: Parsing du fichier json """

# data, fields = parse('data/data.json', 65000)
# data = doublonsNonSpams(data, fields)

""" --- ETAPE 2: Algorithme PageRank et spammicité """

# rg = ReviewGraph(data, fields, window = 1)
# rg.computeScores(niter=50)

""" --- Récupérer les 50 pires produits et 50 meilleurs produits """

# worst = rg.get_k_worst(50)
# best = rg.get_k_best(50)

""" --- Pour afficher la moyenne dans le temps d'un produit et l'histogramme
    --- du nombre d'avis postés par mois, il faut entrer la commande suivante """

# rg.prod_timeline(id_produit)
# Par exemple, id_produit = worst[20], ou id_produit = '9707716436'

""" --- Méthode 1: (cf. explications que j'avais écrites la dernière fois dans 
    --- le fichier svm.ipynb du Github: master/resultats_svm/svm.ipynb) """

""" On souhaite par exemple étudier un des pires produits ('B000O8TWE8').
    Il faut d'abord repérer graphiquement une période de burst """

# rg.prod_timeline('B000O8TWE8')

""" On voit que la période de burst se trouve entre les mois 100 et 140
    Pendant cette période, la moyenne augmente alors qu'elle était à l'origine
    basse. On s'attend donc à ce que les revues dans le burst soient très positives
    contrairement aux revues hors-burst qui devraient être neutres ou négatives """

# Fonction qui fait toute la partie pré-processing des revues sur le produit
# choisi, entre les mois month_min et month_max. On indique la période de burst
# avec month_min = 100 et month_max = 140.

# tp = etude_1(data, fields, 'B000O8TWE8', month_min=0, month_max=99, vtype='tf-idf', analyzer='word', ngram_range=(2,2), display_features=True)

# tp est un objet TextProcessor. On peut récupérer les matrices de similarité
# et autres attributs self (cf le code)

# A indiquer:

# 1) Seuil sur la similarité: seuil à partir duquel on considère que deux revues
# sont très similaires (similarité cosinus), par exemple 0.7.
# Les similarités supérieures ou égales au seuil sont mises à 1 (similaires)
# les autres à 0 (non-similaires)

# 2) Seuil sur la somme des similarités seuillées: on a fait la somme de chaque
# ligne de la matrice de similarité codée en 0/1. On obtient donc 1 valeur par
# ligne, qui correspond pour chaque revue au nombre de revues similaires.
# Si cette somme est supérieure à un seuil k (cad si une revue est similaire
# à plus de k autres revues), on se dit que c'est suspect
# Par exemple ici, on prendra 1

# --> revues non similaires: < k
# --> revues similaires: >= k

""" Explication de l'affichage (dernier graphe) """

# On se retrouve avec un histogramme sur deux classes: les revues non similaires
# dans la période (month_min, month_max) en vert, et les revues similaires
# en violet. 

# On a en fait appliqué un SVM sur les revues avec comme classe -1 si une revue
# est non similaire aux autres et 1 si elle est similaire à d'autres revues.
# Les batons de l'histogramme correspondent
# - en vert: aux mots discriminants pour les revues non similaires 
#            (cad les mots distinguant bien cette classe de l'autre classe)
# - en violet: les mots discriminants pour les revues très similaires

# Dans notre exemple:
# On avait dit que pendant la période de burst (mois 100 à 140), la moyenne augmentait
# alors que la moyenne au départ était basse. On s'attendait alors à ce que les
# revues très similaires (spam) soient plutôt positives, contre des revues 
# négatives voire neutres pour les revues non similaires (dans la période de burst
# mais non-spam).
# Ici, les 5 mots les plus discriminants pour les revues non-spams (vert) sont:
# {'seem work', 'boost signal', 'work better', 'people sell', 'product work'}.
# On voit donc qu dans cette période de temps, les revues non-spam (non similaires)
# sont plutôt neutres
# Les 5 mots les plus discriminants pour les revues très très similaires (violet)
# sont: {'money work', 'cannot tell', 'made differ', 'better recept', 'improv signal'}
# donc 2 positifs, 3 neutres. Ca confirme nos suspicions.

""" Il faut maintenant refaire la même étude sur le même produit en période non burst """ 

# Par exemple month_min=0, month_max=99
# 1er seuil: 0.6
# 2e seuil: 1

# Attributs discriminants:
# Revues non similaires (vert): {'dead zone', 'waste money', 'still servic'}
# Revues similaires (violet): {'drop call', 'booster phone', 'one bar', etc...}
# Dans les deux cas, plutôt négatif (on rappelle que de base la moyenne est plutôt mauvaise)

# --> confirme nos suspicions
