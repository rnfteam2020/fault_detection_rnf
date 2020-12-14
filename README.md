# RNF project: Detekce chyby

#### Zpracovali: 
- Artyom Voronin
- Martin Havelka
- Tomáš Kopčil
- Jan Bolcek
- Jan Hrůzek

## Model a generování dat
Byl použit jednoduchý mechanický oscilátor skládající se z tělěsa, pružiny a tlumiče. Parametry tohoto systému jsou tedy hmostnost *m*, tuhost pružiny *k* a tlumení *b*.
Oscilátor byl buzen silou *F* o průběhu "step" a sinus a měrena byla výchylka a rychlost tělesa.

#### Generování korektních a chybných dat
Pro zmíněný model bylo vygenerováno sto různých kombinací parametrů a pro ně naměřena odezva. Tato data byla označena jako korektní.
Jako chybná byla uvažována situace, kdy je poměrný útlum soustavy větší, nebo roven jedné, tedy soustava je přetlumená a nedochází ke kmitání. Taková data pak byla označena jako chybná.

## Statistické zpracování dat
Pro každý balík naměřených dat byla zpracována statistická analýza. Určeny byly následující statistické parametry:
- minimum
- maximum
- aritmetický průměr
- medián
- standardní odchylka
- rozptyl
- RMS
- Fourierova transformace pomocí FFT algoritmu a následně vybrány 3 nejvíce dominantní frekvence.

Tyto parametry byly zabaleny společně s označením, zda se jedná o chybná, nebo korektní data, a následně použita jako vstup pro neuronovou síť.

## Dataset
Ze statisticky zpracovaných dat byl vytvořen dataset, který odpovídá vstupům neuronové sítě. Jedná se o tensor, který obsahuje hodnoty features (statistické parametry) a labels (označení správných a chybných dat, 1/0). Následně byl dataset rozdělen na trénovací a validační data v poměru 80% ku 20%. Takto rozdělený dataset byl dále použit v neuronové síti.

## Neurová síť
Pro vytvoření neuronové sítě byl použit nástroj PyTorch. Byl vytvořen model s jednou vstupní, skrytou a výstupní vrstvou.  

  #### Velikost vsrtev:
      Vstupní vrstva: 20
      Skrytá vrstva: 16
      Výstupní vrstva: 1

  #### Parametry sítě:
      Aktivační funkce: sigmoid
      Optimizator: Adam 
      Loss function: MSELoss 
      learning rate: 0,05 
      počet epoch: 300 
      velikost batch: 10

Trénování neuronové sítě tedy probíhalo v závislosti na velikosti batch a na počtu epoch. Výsledky trénování jsou zobrazené v následující části. 

## Výsledky


## Instalace 
- install dependencies
```shell
pip install -r requirements.txt
```
- install [PyTorch](https://pytorch.org/get-started/locally/)

## Zdroje
- [wiki](https://en.wikipedia.org/wiki/Fault_detection_and_isolation)
- [PyTorch: nn](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-nn)
- [link](https://www.sciencedirect.com/science/article/pii/S1876610218304831)
- [kaggle](https://www.kaggle.com/c/vsb-power-line-fault-detection/notebooks)
- [FDI](https://www.researchgate.net/publication/221412815_Fault_detection_methods_A_literature_survey/)
