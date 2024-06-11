# Graf neuronske mreže
U ovom radu bavili smo se graf neuronskim mrežama. Implementacija je rađena u Google Colabu koristeći Pytorch library. Radili smo klasifikaciju čvorova te klasifikaciju grafa čija je preciznost uz različite parametre prikazana kroz tablice.
* [Uvod](#uvod)
### Tablica 1. 
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

### Tablica 2. 
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

### Tablica 4. 
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

### Tablica 4. 
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



## Uvod
