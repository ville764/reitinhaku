# reitinhaku

Asennusohje
Lataa koodit

Koodin ajaminen
Poetry run main.py (mikäli tässä tulee ongelmia en osaa kommentoida, poetry on uusi työkalu minulle)
Ohjelma lataa kartat ja ajaa testejä A* ja JPS algoritmeillä
Sen jälkeen se visualisoi testien tulokset
Sulkemalla graafin se siirtyy visualisoimaan A* ja JPS algoritmien toimintaa kartalla
Kartasta seuraavaan voi siirtyä sulkemalla kartan
Ohjelman saa keskeytettyä CTRL-C


Yksikkötestien ajaminen
reitinhakuhakemistossa
poetry run pytest tests --cov=. --cov-report=term-missing
