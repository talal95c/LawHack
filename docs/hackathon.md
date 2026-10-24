# Organisation du hackathon

Ce document rassemble la feuille de route, l'organisation de l'équipe et les consignes de code préparées pour le hackathon.

## Feuille de route

**Hackathon (6–7 h)**

1. Squelette repo, schémas, ingestion PDF, zonage par règles.
2. Client Système 1 (Jev / Codiv) + attribution → registre.
3. Réponse Mistral citée + politique d'honnêteté.
4. UI chat avec pastilles ; écran partagé Astra.
5. Benchmark sur un échantillon d'arrêts + chiffres pour le pitch.

**Après**

- Vérificateur générique de textes rédigés par humain / Harvey / Astra.
- Serveur MCP branchable sur Le Chat, Claude, Legora, Harvey.
- Plusieurs arrêts par session ; dossiers longs non balisés (assignations, conclusions, pièces).
- Restitution des faits : chronologie acteur · action · date · source · statut · confiance.
- Hébergement UE (ex. eu/jev) pour la confidentialité des dossiers.

## Organisation de l'équipe (3 personnes)

Chaque personne possède ses dossiers ; le contrat entre elles est le format du registre (`lawhack/schema.py`), figé tôt.

| Rôle | Périmètre | Dossiers |
|---|---|---|
| **1 · Système 1 & registre** | Zonage robuste (20 arrêts variés), segmentation, prompts Jev EN vs FR, calibration des seuils, bascule Jev ↔ OpenJev | `lawhack/zoning.py`, `segmenter.py`, `attributor.py`, `system_one.py` |
| **2 · Système 2 & produit** | Récupération, réponse Mistral citée `[S-xx]`, vérification Jev + abstention, chat, pastilles, animation | `lawhack/answer.py`, `verify.py`, `app/` |
| **3 · Juriste : données, benchmark, pitch** | Annotation avec la grille de rôles, questions pièges + réponses attendues, benchmark Astra vs Legible, choix du cas réel, pitch | `eval/`, `data/` |

Synchronisations : (1) schéma du registre figé ; (2) test de bout en bout à mi-parcours sur le PDF de démo ; (3) gel du code 1 h avant la fin (corrections, répétition, chiffres).

La grille des rôles et les marqueurs d'un arrêt figurent dans [l'ontologie d'attribution du README](../README.md#5-ontologie-dattribution).

## Consignes pour les agents de code

Ce README est la **source de vérité** du produit. Avant de coder :

- **Respecter le pipeline du §4** et la séparation : règles/code → Jev (décisions typées) → Mistral (rédaction).
- **Ne pas demander à Jev de générer du texte** ni du JSON libre ; uniquement `noul` / `choice` / `score`.
- **Toujours propager** probabilités et confiance jusqu'à la réponse finale (pastilles).
- **Toujours inclure `INDETERMINE`** dans les choix de locuteur.
- Utiliser les **codes de l'ontologie du §5** tels quels ; ne pas en inventer sans mettre à jour ce README.
- Chaque affirmation de la réponse doit pointer vers un segment `S-xx` existant ; sinon, abstention.
- Le jeu labellisé ne sert **qu'à l'évaluation** ; ne jamais l'injecter dans les prompts.
- Aucune clé dans le code ou les commits ; lire la configuration depuis l'environnement.
- Passer par l'abstraction `SystemOneClient` pour pouvoir basculer Jev ↔ OpenJev.
- Ne pas promettre « zéro erreur » dans l'UI ou la documentation.
- Petites PR focalisées, avec tests (`pytest`) sur zonage, segmentation et politique d'honnêteté.
