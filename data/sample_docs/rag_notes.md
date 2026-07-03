# Qu'est-ce que le Retrieval-Augmented Generation (RAG)

Le RAG est une architecture qui combine un système de recherche
d'information avec un modèle de langage génératif. Plutôt que de se fier
uniquement aux connaissances mémorisées pendant l'entraînement du modèle,
le système récupère des passages pertinents dans une base de documents
externe, puis les fournit en contexte au LLM pour générer une réponse.

## Pourquoi utiliser le RAG plutôt qu'un fine-tuning ?

Le fine-tuning modifie les poids du modèle pour lui apprendre de nouvelles
connaissances, ce qui est coûteux et doit être refait à chaque mise à jour
des données. Le RAG, à l'inverse, sépare la base de connaissances du
modèle : on peut mettre à jour les documents indexés sans jamais
réentraîner le LLM. Cela réduit aussi les hallucinations, puisque le
modèle peut citer ses sources et s'appuyer sur un contexte vérifiable.

## Le chunking

Le chunking consiste à découper les documents en passages de taille
raisonnable avant de les indexer. Un chunk trop grand dilue le signal
pertinent parmi du bruit ; un chunk trop petit perd le contexte
nécessaire à la compréhension. Les stratégies avancées de chunking
utilisent des frontières sémantiques (changement de sujet détecté par
similarité d'embeddings) plutôt qu'un découpage arbitraire par nombre de
caractères.

## Le retrieval hybride

Le retrieval dense, basé sur des embeddings, capture la similarité
sémantique mais peut manquer des correspondances lexicales exactes, comme
un identifiant de produit ou un acronyme rare. Le retrieval lexical BM25,
à l'inverse, excelle sur les termes exacts mais ne comprend pas les
synonymes ni les reformulations. Combiner les deux approches, par exemple
via une fusion de rangs (Reciprocal Rank Fusion), permet de bénéficier des
forces des deux méthodes.

## Le reranking

Après une première récupération de candidats (par exemple 20 à 40
passages), un reranker de type cross-encoder réévalue chaque paire
(question, passage) avec un modèle qui traite les deux textes ensemble,
ce qui donne un score de pertinence beaucoup plus précis qu'un simple
produit scalaire entre deux embeddings calculés séparément. Cette étape
est plus coûteuse en calcul, donc on ne l'applique qu'à un petit nombre de
candidats déjà pré-filtrés.

## L'évaluation d'un système RAG

Un système RAG se mesure sur deux axes distincts : la qualité du
retrieval (les bons passages sont-ils récupérés ?) et la qualité de la
génération (la réponse produite est-elle fidèle au contexte et pertinente
par rapport à la question ?). Des métriques comme la faithfulness
(fidélité) mesurent si chaque affirmation de la réponse est bien
supportée par le contexte fourni, ce qui permet de détecter les
hallucinations de façon quantitative plutôt qu'au jugement subjectif.
