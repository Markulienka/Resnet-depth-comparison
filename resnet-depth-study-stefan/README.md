# Súboj generácií ResNet: Vplyv hĺbky na stabilitu učenia

## 1. Cieľ projektu

Tento projekt porovnáva dve reziduálne konvolučné neurónové siete: **ResNet-34** a **ResNet-50**.  
Hlavným cieľom je zistiť, ako sa zmena hĺbky siete prejaví na:

- stabilite učenia,
- rýchlosti konvergencie,
- pamäťovej náročnosti,
- dosiahnutej presnosti klasifikácie.

Projekt je navrhnutý ako **jednoduchý študentský experiment**, bez zbytočne komplikovanej implementácie.  
Namiesto implementácie architektúr od nuly sa použijú hotové modely z knižnice **torchvision**.

---

## 2. Výskumná otázka

Prečo sa hlbšie siete učia ťažšie a ako skip connections v architektúre ResNet pomáhajú tento problém zmierniť?

---

## 3. Stručné teoretické pozadie

Pri veľmi hlbokých neurónových sieťach sa často objavuje problém **miznúceho gradientu**.  
To znamená, že pri spätnom šírení chyby sa gradient v skorších vrstvách zmenšuje natoľko, že váhy sa aktualizujú veľmi pomaly alebo vôbec.

Architektúra **ResNet** tento problém rieši pomocou **reziduálnych prepojení** (*skip connections*), ktoré umožňujú prenášať vstup priamo cez blok:

\[
y = F(x) + x
\]

Takéto prepojenie pomáha:

- lepšiemu toku gradientu,
- stabilnejšiemu trénovaniu,
- efektívnejšiemu učeniu hlbších sietí.

---

## 4. Porovnávané modely

### ResNet-34
- plytšia architektúra,
- používa **basic residual blocks**,
- nižšia pamäťová náročnosť,
- rýchlejšie trénovanie.

### ResNet-50
- hlbšia architektúra,
- používa **bottleneck residual blocks**,
- vyššia výpočtová aj pamäťová náročnosť,
- potenciálne lepšia schopnosť extrakcie príznakov.

---

## 5. Dataset

Použitý dataset: **CIFAR-10**

Obsahuje:
- 60 000 obrázkov,
- 10 tried objektov,
- obrázky veľkosti 32x32 pixelov.

Rozdelenie:
- 50 000 trénovacích vzoriek,
- 10 000 testovacích vzoriek.

Oficiálny zdroj:
- Dataset: <https://www.cs.toronto.edu/~kriz/cifar.html>

---

## 6. Jednoduchý plán realizácie

1. Načítať dataset CIFAR-10.
2. Pripraviť transformácie a normalizáciu dát.
3. Načítať model ResNet-34.
4. Natrénovať model a uložiť metriky.
5. Načítať model ResNet-50.
6. Natrénovať model a uložiť metriky.
7. Porovnať výsledky pomocou grafov a tabuliek.
8. Vyhodnotiť:
   - loss,
   - accuracy,
   - gradient norm,
   - training time,
   - memory usage.

---

## 7. Čo sa bude sledovať

Počas trénovania oboch modelov sa budú zaznamenávať tieto metriky:

- **train loss**
- **train accuracy**
- **validation/test accuracy**
- **validation/test loss**
- **čas jednej epochy**
- **celkový čas trénovania**
- **gradient norm**
- **pamäťová náročnosť**

Tieto metriky umožnia porovnať:
- stabilitu učenia,
- rýchlosť konvergencie,
- efektivitu modelov.

---

## 8. Očakávané výsledky

Predpokladá sa, že:

- **ResNet-50** dosiahne podobnú alebo mierne vyššiu presnosť,
- **ResNet-50** bude trénovaný pomalšie,
- **ResNet-50** spotrebuje viac pamäte,
- skip connections prispejú k stabilnejšiemu učeniu aj pri väčšej hĺbke.

Dôležité však je, že hlbší model nemusí byť pri jednoduchom študentskom experimente automaticky výrazne lepší.  
Pri menšom datasete a kratšom trénovaní môže byť **ResNet-34 praktickejšia voľba**.

---

## 9. Navrhovaná štruktúra projektu

```text
resnet-depth-study/
│
├── src/
│   ├── config.py
│   ├── models.py
│   ├── utils.py
│   └── train.py
│
├── results/
│   ├── resnet34_history.csv
│   ├── resnet50_history.csv
│   └── best_models/
│
├── report/
│   ├── main.tex
│   ├── images/
│   │   ├── loss_curve.png
│   │   ├── accuracy_curve.png
│   │   ├── gradient_curve.png
│   │   └── memory_usage.png
│   └── references.bib
│
├── notebook/
│   └── analysis.ipynb
│
├── requirements.txt
├── README.md
└── project_structure.md
```

---

## 10. Úloha jednotlivých Python súborov

### `config.py`
Obsahuje základné nastavenia projektu:
- názov modelu,
- počet epoch,
- learning rate,
- batch size,
- paths,
- device (CPU/GPU).

### `models.py`
Obsahuje funkciu na výber modelu:
- ResNet-34,
- ResNet-50.

Modely sa načítavajú z `torchvision.models`.

### `utils.py`
Pomocné funkcie:
- nastavenie seed,
- výpočet accuracy,
- výpočet gradient norm,
- uloženie histórie do CSV,
- vytvorenie priečinkov.

### `train.py`
Hlavný tréningový skript:
- načítanie dát,
- tréning jednej epochy,
- validácia,
- záznam metrík,
- uloženie najlepšieho modelu,
- export výsledkov.

---

## 11. Najjednoduchšia metodika do práce

V metodike môžeš použiť napríklad tento jednoduchý opis:

> V experimente boli porovnané architektúry ResNet-34 a ResNet-50 na datasete CIFAR-10.  
> Obe siete boli implementované pomocou knižnice torchvision v prostredí PyTorch.  
> Modely boli trénované za rovnakých podmienok, aby bolo možné spravodlivo porovnať ich správanie.  
> Počas trénovania sa sledovali hodnoty straty, presnosti, gradient normy, času a pamäťovej náročnosti.  
> Cieľom bolo zistiť, ako sa zvýšenie hĺbky siete prejaví na stabilite učenia a efektivite trénovania.

---

## 12. Čo napísať do výsledkov

Vo výsledkovej časti sa sústreď na tieto body:

- Ktorý model konvergoval rýchlejšie?
- Ktorý model dosiahol vyššiu presnosť?
- Bol priebeh loss stabilný?
- Rástla pamäťová náročnosť pri hlbšom modeli?
- Naznačujú gradienty stabilnejšie učenie pri skip connections?

---

## 13. Záver projektu

Záver môže stáť na tejto logike:

- ResNet ukazuje, že skip connections pomáhajú trénovať hlbšie siete.
- Väčšia hĺbka môže priniesť lepšie výsledky, ale za cenu vyššej výpočtovej a pamäťovej náročnosti.
- Pre jednoduchý experiment na CIFAR-10 nemusí byť hlbší model vždy výrazne výhodnejší.
- Praktický prínos projektu spočíva v pochopení kompromisu medzi hĺbkou siete a efektivitou učenia.

---

## 14. Dôležité odporúčanie

Tento projekt nemá byť extrémne komplikovaný.  
Najrozumnejší prístup je:

- použiť hotové modely z `torchvision`,
- urobiť čisté porovnanie,
- vytvoriť grafy,
- stručne a vecne interpretovať výsledky.

To úplne stačí na dobrý študentský projekt.
