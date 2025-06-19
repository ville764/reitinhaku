# reitinhaku

Asennusohje
Lataa koodit

Koodin ajaminen
siirry hakemistoon reitinhaku
anna terminaalissa: Poetry install
anna terminaalissa: Poetry run python main.py 
Ohjelma lataa kartat ja ajaa testejä A* ja JPS algoritmeillä
Sen jälkeen se visualisoi testien tulokset tulosteeseen ja graafeiksi
Sulkemalla graafin se siirtyy visualisoimaan A* ja JPS algoritmien toimintaa kartalla
Kartasta seuraavaan voi siirtyä sulkemalla kartan
Ohjelman saa keskeytettyä CTRL-C


Yksikkötestien ajaminen
reitinhakuhakemistossa
poetry run pytest tests --cov=. --cov-report=term-missing
