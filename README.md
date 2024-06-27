# Graf neuronske mreže
![](gnnslika.webp)

* [Uvod](#uvod)
* [Tablice](#tablice)
  * [Tablica 1.](#tablica1)
  * [Tablica 2.](#tablica2)
  * [Tablica 3.](#tablica3)
  * [Tablica 4.](#tablica4)
* [Projekt](#Projekt)
* [Kod](#Kod)
## Uvod
Za početak ćemo definirati što je to zapravo graf. Grafovi su sveprisutna struktura podataka i univerzalni jezik za opisivanje
složenih sustava. Općenito, možemo reći da je to skup objekata, odnosno čvorova, zajedno s njihovim interakcijama, odnosno bridovima. Oni nude matematičku osnovu koju možemo nadograditi kako bi mogli analizirati, razumjeti te učiti iz složenih sustava stvarnog svijeta. U zadnjih dvadeset i pet
godina, došlo je do dramatičnog povećanja količine i kvalitete grafički strukturiranih podataka koji su dostupni istraživačima. Izazov je otključati potencijal tog mnoštva podataka. Kao što smo već spomenuli, graf $G = (V, E)$ je definiran skupom čvorova $V$ i skupom bridova $E$ između tih čvorova. Brid koji ide od čvora $u$ do čvora $v$ označavamo s $(u,v)$. Graf može imati usmjerene ili neusmjerene bridove. Prikladan način za predstavljanje grafova jest matrica susjedstva 
$A$ $\in$ $\mathbb{R}$<sup>$(|V| * |V|)$</sup>. Tada svaki čvor indeksira određeni redak i stupac u matrici susjedstva. Na taj način možemo prisutnost brida između 2 čvora predstaviti jedinicom, a odsutnost brida nulom. Odnosno, ako postoji brid između čvora $u$ i $v$ onda vrijedi $A[u,v] = 1$, a u protivnom vrijedi $A[u,v] = 0$.
## Projekt
Cilj ovog projekta je klasifikacija čvorova grafa te klasifikacija samog grafa korištenjem graf neuronskih mreža. U projektu je korištena biblioteka PyTorch Geometric. Implementirano je nekoliko modela čiji su rezultati prikazani u tablicama. U modelima smo koristili različite konvolucijske slojeve, te različite parametre (learning rate te weight decay), pri čemu smo dobivali različite rezultate ovisno o svim tim svojstvima.

## Kod
Za početak projekta smo prvo morali instalirati torch-geometric library te importati sve potrebne stvari koje će nam poslužiti prilikom stvaranja modela.

```python
!pip install torch-geometric
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv
)
```

Prvo smo radili Node Classification u kojem smo učitali CiteSeer dataset. To je skup podataka koji se sastoji od 3312 znanstvenih publikacija razvrstanih u jednu od šest klasa. Citatnu mrežu čine 4732 poveznice. Svaka objava u skupu podataka opisana je vektorom riječi s vrijednosti 0/1 koji označava odsutnost/prisutnost odgovarajuće riječi iz rječnika. Rječnik se sastoji od 3703 jedinstvene riječi. 

```python
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
```
Nakon učitavanja skupa podataka smo još ispisali neke osnovne stvari koje su dio tog skupa.

![](NodeClassification.png)

Nakon tog smo krenuli u implementiranje prvog modela. Koristili smo GCNConv layer te ReLU kao aktivacijsku funkciju. Na kraju smo ispisali softmax distribuciju preko broja klasa.

```python
class GCNNode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
```


Iza tog smo trenirali model na trening podacima za 11 epoha pri čemu smo mijenjali svojstva (learning_rate i weight_decay). Izračunali smo rezultate, a zatim ih i ispisali.


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNNode().to(device)
print(model)
data = dataset[0].to(device)

learning_rates = [0.1,0.01,0.001,0.0001]
weight_decays = [0.1,0.01,0.001,0.0001]
for learning_rate in learning_rates:
  for weight_decay in weight_decays:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    model.train()
    print(f'Learning_rate: {learning_rate}, weight_decay: {weight_decay}')
    print('=============================================================')
    print()
    for epoch in range(1, 12):
      optimizer.zero_grad()
      out = model(data)
      loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()
      model.eval()
      pred = model(data).argmax(dim=1)
      correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
      test_acc = int(correct) / int(data.test_mask.sum())
      print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}')
```

Nakon što smo završili s prvim modelom, napravili smo drugi u kom smo koristili GraphConv layer te kao u prošlom primjeru ReLU aktivacijsku funkciju. Na kraju smo ispisali softmax distribuciju preko broja klasa.

```
class GraphNode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(dataset.num_node_features, 16)
        self.conv2 = GraphConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
```

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

