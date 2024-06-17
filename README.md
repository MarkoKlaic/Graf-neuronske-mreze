# Graf neuronske mreže
![](gnnslika.webp)

U ovom radu bavili smo se graf neuronskim mrežama. Implementacija je rađena u Google Colabu koristeći Pytorch library. Radili smo klasifikaciju čvorova te klasifikaciju grafa čija je preciznost uz različite parametre prikazana kroz tablice.
* [Uvod](#uvod)
* [Tablice](#tablice)
  * [Tablica 1.](#tablica1)
  * [Tablica 2.](#tablica2)
  * [Tablica 3.](#tablica3)
  * [Tablica 4.](#tablica4)
    
## Uvod
Za početak ćemo definirati što je to zapravo graf. Grafovi su sveprisutna struktura podataka i univerzalni jezik za opisivanje
složenih sustava. Općenito, možemo reći da je to skup objekata, odnosno čvorova, zajedno s njihovim interakcijama, odnosno bridovima. Oni nude matematičku osnovu koju možemo nadograditi kako bi mogli analizirati, razumjeti te učiti iz složenih sustava stvarnog svijeta. U zadnjih dvadeset i pet
godina, došlo je do dramatičnog povećanja količine i kvalitete grafički strukturiranih podataka koji su dostupni istraživačima. Izazov je otključati potencijal tog mnoštva podataka. Kao što smo već spomenuli, graf $G = (V, E)$ je definiran skupom čvorova $V$ i skupom bridova $E$ između tih čvorova. Brid koji ide od čvora $u$ do čvora $v$ označavamo s $(u,v)$. Graf može imati usmjerene ili neusmjerene bridove. Prikladan način za predstavljanje grafova jest matrica susjedstva 
$A$ $\in$ $R$<sup>$(|V| * |V|)$</sup>. Tada svaki čvor indeksira određeni redak i stupac u matrici susjedstva. Na taj način možemo prisutnost brida između 2 čvora predstaviti jedinicom, a odsutnost brida nulom. Odnosno, ako postoji brid između čvora $u$ i $v$ onda vrijedi $A[u,v] = 1$, a u protivnom vrijedi $A[u,v] = 0$.
## Tablice
### <ins>Tablica 1. <a class="anchor" id="tablica1"></a></ins> 
## GCNConv za klasifikaciju čvorova s obzirom na različite hiperparametre (11 epoha)
| *Learning rate*   | *Weight decay*   | *Accuracy* |
| -------------   | ------   | -------- |
| 0.1             | 0.1      | 0.7250   |
| 0.1             | 0.01      | 0.6910   |
| 0.1             | 0.001      | 0.5740   |
| 0.1             | 0.0001      | 0.6230   |
| 0.01            | 0.1      | 0.5650   |
| 0.01            | 0.01      | 0.6140   |
| 0.01            | 0.001      | 0.6620   |
| 0.01            | 0.0001      | 0.6670   |
| 0.001            | 0.1      | 0.6670   |
| 0.001            | 0.01      | 0.6650   |
| 0.001            | 0.001      | 0.6750   |
| 0.001            | 0.0001      | 0.6760   |
| 0.0001            | 0.1      | 0.6770   |
| 0.0001            | 0.01      | 0.6780   |
| 0.0001            | 0.001      | 0.6780   |
| 0.0001            | 0.0001      | 0.6780   |

### <ins>Tablica 2. <a class="anchor" id="tablica2"></a></ins> 
## GraphConv za klasifikaciju čvorova s obzirom na različite hiperparametre (11 epoha)
| *Learning rate*   | *Weight decay*   | *Accuracy* |
| -------------   | ------   | -------- |
| 0.1             | 0.1      | 0.4680   |
| 0.1             | 0.01      | 0.4020   |
| 0.1             | 0.001      | 0.4790   |
| 0.1             | 0.0001      | 0.5330   |
| 0.01            | 0.1      | 0.5410   |
| 0.01            | 0.01      | 0.5490   |
| 0.01            | 0.001      | 0.5720   |
| 0.01            | 0.0001      | 0.6330   |
| 0.001            | 0.1      | 0.6300  |
| 0.001            | 0.01      | 0.6290   |
| 0.001            | 0.001      | 0.6320   |
| 0.001            | 0.0001      | 0.6330   |
| 0.0001            | 0.1      | 0.6320   |
| 0.0001            | 0.01      | 0.6310   |
| 0.0001            | 0.001      | 0.6290   |
| 0.0001            | 0.0001      | 0.6340   |

### <ins>Tablica 3. <a class="anchor" id="tablica3"></a></ins>
## GCNConv za klasifikaciju grafa s obzirom na različite hiperparametre (22 epohe)
| *Learning rate*   | *Weight decay*   | *Accuracy* |
| -------------   | ------   | -------- |
| 0.1             | 0.1      | 0.7368   |
| 0.1             | 0.01      | 0.7368   |
| 0.1             | 0.001      | 0.8684   |
| 0.1             | 0.0001      | 0.7368   |
| 0.01            | 0.1      | 0.7368   |
| 0.01            | 0.01      | 0.8684   |
| 0.01            | 0.001      | 0.8684   |
| 0.01            | 0.0001      | 0.8421   |
| 0.001            | 0.1      | 0.8158  |
| 0.001            | 0.01      | 0.8421   |
| 0.001            | 0.001      | 0.8684   |
| 0.001            | 0.0001      | 0.7895   |
| 0.0001            | 0.1      | 0.7895   |
| 0.0001            | 0.01      | 0.7895   |
| 0.0001            | 0.001      | 0.7895   |
| 0.0001            | 0.0001      | 0.7895   |

### <ins>Tablica 4. <a class="anchor" id="tablica4"></a></ins> 
## GraphConv za klasifikaciju grafa s obzirom na različite hiperparametre (22 epohe)
| *Learning rate*   | *Weight decay*   | *Accuracy* |
| -------------   | ------   | -------- |
| 0.1             | 0.1      | 0.7368   |
| 0.1             | 0.01      | 0.7368   |
| 0.1             | 0.001      | 0.7368   |
| 0.1             | 0.0001      | 0.7368   |
| 0.01            | 0.1      | 0.7368   |
| 0.01            | 0.01      | 0.7368   |
| 0.01            | 0.001      | 0.7368   |
| 0.01            | 0.0001      | 0.8158   |
| 0.001            | 0.1      | 0.7895  |
| 0.001            | 0.01      | 0.7895   |
| 0.001            | 0.001      | 0.7895   |
| 0.001            | 0.0001      | 0.7895   |
| 0.0001            | 0.1      | 0.7895   |
| 0.0001            | 0.01      | 0.7895   |
| 0.0001            | 0.001      | 0.7895   |
| 0.0001            | 0.0001      | 0.7895   |

