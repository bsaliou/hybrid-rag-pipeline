# Pourquoi ces choix techniques ?

Ce document justifie chaque décision d'architecture non triviale du projet.
L'objectif n'est pas d'utiliser la stack "à la mode", mais de montrer un
raisonnement : quel problème chaque choix résout, quelles alternatives ont
été écartées, et quel compromis a été accepté.

---

## 1. Chunking sémantique plutôt qu'un split fixe (500 tokens)

**Le problème avec un split fixe** : il coupe le texte sans regarder le
sens. Une idée peut être scindée entre deux chunks, ce qui casse le
contexte au moment du retrieval (un chunk isolé ne "veut plus rien dire")
et dégrade la qualité de la génération finale.

**L'approche retenue** : segmenter par phrases, calculer l'embedding de
chaque phrase, puis détecter les frontières où la similarité cosinus entre
phrases consécutives chute sous un seuil — signe d'un changement de sujet.
Des garde-fous de taille (min/max tokens) évitent les micro-chunks
inexploitables et les chunks trop gros pour le reranker.

**Alternative écartée** : un vrai topic modeling (LDA, clustering
d'embeddings sur tout le document) est plus rigoureux mais complexifie le
pipeline pour un gain marginal sur des documents de taille modeste. Le
seuil de similarité est une heuristique simple, explicable, et suffisante
pour la majorité des cas d'usage documentaire.

**Compromis assumé** : le seuil (`semantic_similarity_threshold`) est fixe
et pas appris — sur un corpus très hétérogène, il faudrait le calibrer par
domaine plutôt que garder une valeur unique.

---

## 2. Retrieval hybride (dense + BM25) plutôt que dense seul

**Le problème avec le dense seul** : les embeddings capturent bien la
similarité *sémantique* mais lissent parfois l'information lexicale
exacte. Une requête contenant un identifiant précis (référence produit,
nom propre rare, numéro de version, acronyme métier) peut ne pas remonter
le bon passage si ce terme n'a pas de représentation sémantique forte dans
l'espace d'embedding — deux termes proches en sens mais différents dans la
forme se retrouvent proches, ce qui peut noyer un match exact.

**Le problème avec BM25 seul** : c'est un modèle purement lexical — il ne
comprend ni synonymes, ni reformulations, ni paraphrases. Une requête
formulée différemment du document source peut totalement rater le bon
passage.

**La solution** : faire tourner les deux retrievers en parallèle et
fusionner leurs résultats par **Reciprocal Rank Fusion (RRF)** :

```
RRF(d) = Σ 1 / (k + rank_i(d))
```

RRF ne combine que les *rangs*, pas les scores bruts — ce qui évite le
problème de calibration entre une similarité cosinus (bornée [0,1]) et un
score BM25 (non borné, dépendant du corpus). C'est la méthode utilisée en
production par Elasticsearch et Weaviate pour cette même raison.

**Alternative écartée** : pondération linéaire `α·score_dense +
(1-α)·score_bm25`. Plus simple à comprendre mais fragile : le score BM25
change d'échelle selon la taille du corpus, ce qui oblige à re-calibrer
`α` à chaque changement de dataset. RRF est insensible à cette dérive.

---

## 3. Reranking cross-encoder après le retrieval hybride

**Pourquoi une deuxième passe ?** Le retrieval (dense ou BM25) est un
*bi-encoder* : la requête et le document sont encodés **séparément**, puis
comparés par un simple produit scalaire. C'est rapide (indexable à
l'échelle du million de documents) mais imprécis, car le modèle ne voit
jamais la requête et le document *ensemble*.

Un **cross-encoder** prend la paire (requête, document) en entrée unique
et produit un score d'interaction fine entre les deux textes — beaucoup
plus précis, mais impossible à faire tourner sur tout un corpus (il faut
une inférence par paire, donc O(N) appels modèle par requête).

**Le compromis retrieve-then-rerank** : on utilise le retrieval hybride
(rapide) pour réduire le corpus à ~20-40 candidats, puis on applique le
cross-encoder (précis) uniquement sur ce sous-ensemble. On obtient un bon
rappel (peu de faux négatifs à l'étape 1) et une bonne précision (le
cross-encoder trie finement à l'étape 2) à un coût de calcul maîtrisé.

---

## 4. Deux backends pour l'embedding et le reranking (local / API)

Le projet tourne **entièrement gratuitement en local par défaut**
(`bge-large-en-v1.5` pour les embeddings, `ms-marco-MiniLM` pour le
reranking) — aucune clé API n'est nécessaire pour indexer et faire du
retrieval. Une bascule vers OpenAI / Cohere est disponible via variables
d'environnement pour comparer la qualité, sans changer une ligne de code
métier (le contrat d'interface — `embed_documents`, `embed_query`,
`rerank` — reste identique).

C'est un choix pragmatique de portfolio : ça permet à n'importe qui de
cloner le repo et de le faire tourner sans dépenser un centime, tout en
montrant que l'architecture n'est pas verrouillée à un provider.

---

## 5. Chroma plutôt que Qdrant pour la base vectorielle

Qdrant est souvent cité comme référence en production (filtre par
métadonnées avancé, quantification, clustering distribué), mais nécessite
de faire tourner un serveur (Docker) — friction inutile pour un portfolio
qu'un recruteur doit pouvoir lancer en une commande. Chroma est
**persistant sur disque, embarqué, zéro infra**, avec la même logique de
recherche par similarité. Le wrapper (`src/vectorstore/store.py`) expose
une interface volontairement minimale (`add_chunks` / `query`) pour que
remplacer Chroma par Qdrant se limite à réécrire cette seule classe.

---

## 6. Évaluation : mini-framework maison + RAGAS en option

**Pourquoi ne pas se contenter de RAGAS ?** RAGAS est une bonne référence,
mais l'utiliser en boîte noire ne prouve pas qu'on comprend ce que les
métriques mesurent. Le `SimpleEvaluator` maison (`src/evaluation/evaluator.py`)
réimplémente la logique de deux métriques RAGAS clés pour le montrer
explicitement :

- **Faithfulness** : décomposer la réponse générée en affirmations
  atomiques, puis demander à un LLM juge si chaque affirmation est
  supportée par le contexte fourni. Le score est la fraction
  d'affirmations supportées — une mesure directe du taux d'hallucination.
- **Context precision** : pour chaque passage retourné par le retrieval,
  demander à un LLM juge s'il est effectivement utile pour répondre à la
  question. Mesure le bruit dans les résultats du retrieval.
- **Context recall** : décomposer la réponse *de référence* (golden
  answer) en affirmations, et vérifier combien sont couvertes par les
  passages retournés. Mesure les trous du retrieval.
- **Answer relevancy** : reconstruire des questions hypothétiques à partir
  de la réponse générée (technique reprise de RAGAS) et mesurer leur
  similarité embedding avec la question originale — sans appel LLM
  supplémentaire, donc rapide à faire tourner sur de gros jeux de test.

RAGAS reste disponible (`scripts/run_eval.py --use-ragas`) pour comparer
les deux implémentations sur le même golden set — utile pour valider que
le framework maison donne des résultats cohérents avec la référence.

---

## Ce que je ferais différemment en production

- Chunking : calibrer le seuil de similarité par type de document plutôt
  qu'une valeur globale, et logger le taux de "sur-fragmentation" pour
  ajuster automatiquement.
- Retrieval : ajouter un filtre de métadonnées (date, type de document)
  en amont du retrieval sémantique pour les corpus multi-sources.
- Évaluation : construire un golden set plus large (50-100 questions) avec
  double annotation humaine pour calibrer la fiabilité du LLM-juge.
- Observabilité : tracer chaque requête (latence par étage du pipeline,
  chunks retournés, score de confiance) pour du monitoring en continu,
  pas seulement de l'évaluation batch hors-ligne.
