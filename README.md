# DD2424_project


# Composition du dossier


## POP909

Base de données trouvée sur le web, avec du code. Utilisée pour se familiariser avec les librairies et récupérer le fichier processor.py mais rien de plus

## Data
Fichier MIDI classé par artiste

## Modified data
Fichiers MIDI avec sélections des pistes qui semblent pertinentes (les autres ont été supprimés)

## Generated MIDI
Fichier MIDI généré. Le nom donne une indication de la provenance du fichier

## Generated data
Fichier 1/0 des musiques générées

## Encoded data
J'ai essayé de gagner du temps en encodant en amont les données en array 0/1. Il s'avère que c'est trop volumineux et que ça ralentit

## Code
### RNN_file
Contient les fonctions pour l'apprentissage via un RNN semblable à l'Assignement 4 mais modifié pour les spécificités de la musique: softmax -> sigmoid et nouvelle loss function
<br>
On met une sigmoid pour avoir un output entre 0 et 1. Ce qui donne une probabilité pour chaque note d'être jouée.

## scrap
webscraping pour collecter les données

## transform
Fonction pour passer d'un format à l'autre (csv et MIDI)

## processing
Apprentissage avec RNN

## LSTM_pytorch
Apprentissage avec LSTM et usage de pytorch