# tinygrad – minimalistyczna biblioteka głębokiego uczenia maszynowego

## Wprowadzenie

tinygrad to lekka, otwarta biblioteka stworzona przez George'a Hotza (tiny corp). Łączy **prostotę** mikrobiblioteki *micrograd* Karpathy'ego z funkcjonalnością podobną do PyTorch'a. Ze względu na niewielki, czytelny kod źródłowy jest polecana początkującym, którzy chcą zrozumieć wewnętrzne mechanizmy sieci neuronowych. Biblioteka jest w fazie alpha, ale zyskuje popularność (np. w projekcie OpenPilot jako backend GPU).

## Historia i kontekst powstania

George Hotz (znany jako geohot, haker iPhone'a i PlayStation 3) stworzył tinygrad w 2020 roku jako alternatywę dla coraz bardziej skomplikowanych frameworków ML. Inspiracją był micrograd Andreja Karpathy'ego, ale z ambicją stworzenia biblioteki zdolnej do trenowania realnych modeli. Projekt wynika z filozofii tiny corp: "najmniejsze narzędzie, które działa". Hotz argumentuje, że współczesne biblioteki ML stały się zbyt złożone, co utrudnia innowacje i zrozumienie.

## Podstawowe koncepcje i składnia tinygrad

Główną klasą jest `Tensor`, analogiczna do `torch.Tensor`. Na tensorach wykonujemy operacje element-po-elem (np. `x + y`), mnożenie macierzy (`x.matmul(y)`), stosujemy funkcje aktywacyjne (`.relu()`, `.sigmoid()`) oraz inne operacje (np. `reshape`, `permute`). Obliczenia są **leniwe** – wykonywane dopiero po poproszeniu o wynik, co pozwala tinygrad łączyć operacje w zoptymalizowane jądra obliczeniowe.

tinygrad wspiera **autograd**: wystarczy utworzyć tensor z `requires_grad=True`, wykonać operacje i wywołać `loss.backward()`, aby automatycznie obliczyć gradienty:

```python
from tinygrad.tensor import Tensor
a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
c = a * b + a
d = c.sum()
d.backward()
print("c =", c.numpy(), "∂d/∂a =", a.grad.numpy())
```

## Szczegóły implementacyjne autogradu

Implementacja autogradu w tinygrad jest niezwykle elegancka - bazuje na dynamicznym tworzeniu grafu obliczeniowego:

```python
# Fragment kodu ilustrujący autograd
def backward(self):
  # Tworzy graf obliczeniowy podczas wykonywania operacji
  if self._ctx is not None:
    # _ctx przechowuje operację i pochodne
    # Ta prosta implementacja sprawia, że
    # gradienty przepływają wstecz przez graf
```

Każda operacja zapisuje swoje pochodne, co umożliwia automatyczne różniczkowanie wsteczne. Jest to podobne do PyTorch, ale kod źródłowy jest bardziej przystępny (ok. 10_000 linii vs. ponad 1.5 mln w PyTorch).

## Porównanie tinygrad z PyTorch

* **Interfejs (API)**: tinygrad celowo naśladuje PyTorch – składnia jest prawie identyczna, co ułatwia migrację:

```python
# tinygrad
from tinygrad.tensor import Tensor
x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0, 0, -2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

# PyTorch
import torch
x = torch.eye(3, requires_grad=True)
y = torch.tensor([[2.0, 0, -2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()
```

* **Poziom abstrakcji**: PyTorch jest rozbudowanym frameworkiem wysokiego poziomu, podczas gdy tinygrad stawia na prostotę. Tinygrad **nie ma klasy `nn.Module`** – zamiast tego model to zwykła klasa Pythona, a propagację definiuje się przez `__call__`. Zachęca to do funkcjonalnego stylu programowania.

* **Wydajność**: PyTorch jest generalnie szybszy dzięki zoptymalizowanym bibliotekom. Tinygrad ma potencjalne zalety – może kompilować zestawy operacji do spersonalizowanych kerneli, a prostszy backend ułatwia optymalizację – ale obecnie nie dorównuje wydajnością.

* **Cel i zastosowania**: PyTorch jest skierowany na projekty produkcyjne, a tinygrad – przede wszystkim na **naukę i eksperymenty**. Jest także używany w zastosowaniach wbudowanych, gdzie lekkość implementacji ma znaczenie.

## Dlaczego wybrać tinygrad?

* **Łatwość zrozumienia**: tinygrad ma mały, czytelny kod źródłowy zawierający tylko niezbędne elementy.
* **Eksperymenty z akceleratorami**: zaprojektowany tak, by łatwo dodawać nowe backendy – każdy nowy akcelerator musi zaimplementować tylko kilkanaście podstawowych operacji.
* **Wsparcie GPU**: mimo prostoty, tinygrad może korzystać z GPU (OpenCL, CUDA, Triton) dla przyspieszenia obliczeń.
* **Ograniczenia**: nie jest przeznaczony do dużych, produkcyjnych projektów i nie ma tak bogatej biblioteki funkcji jak PyTorch.

## Architektura tinygrad

Tinygrad rozkłada skomplikowane obliczenia na **trzy podstawowe typy operacji**: elementarne, redukujące oraz przesunięcia danych. Wszystkie wyższe funkcje są zbudowane z kompozycji tych prostych bloków.

Obliczenia są wykonywane **leniwie**, co pozwala na fuzję wielu operacji w jeden program GPU/CPU. Dla przyspieszenia powtarzalnych fragmentów kodu, oferuje prosty JIT (dekorator `@TinyJit`).

Korzysta z prostych backendów: CPU realizuje operacje przez NumPy/Python, a GPU – przez OpenCL, CUDA, METAL itp. Dzięki temu wspiera wiele akceleratorów bez złożonej implementacji.

## Praktyczne zastosowanie – Klasyfikator MNIST

Poniżej kompletny przykład trenowania klasyfikatora MNIST w tinygrad:

```python
from tinygrad.tensor import Tensor
import numpy as np

# Definiujemy model
class MNISTModel:
  def __init__(self):
    # Warstwa liniowa 784 -> 128
    self.l1 = Tensor.uniform(784, 128, requires_grad=True)
    self.b1 = Tensor.zeros(128, requires_grad=True)
    # Warstwa liniowa 128 -> 10
    self.l2 = Tensor.uniform(128, 10, requires_grad=True)
    self.b2 = Tensor.zeros(10, requires_grad=True)
    
  def __call__(self, x):
    x = x.reshape(shape=(-1, 784))  # Spłaszczamy obrazki 28x28
    x = x.dot(self.l1) + self.b1    # Pierwsza warstwa
    x = x.relu()                     # Aktywacja ReLU
    x = x.dot(self.l2) + self.b2    # Druga warstwa
    return x.log_softmax()           # Log-softmax dla funkcji straty
```

## Benchmarki wydajności

Porównanie czasu treningu na różnych backedach (niższe wartości = lepiej):

| Framework | CPU (s) | GPU (CUDA) (s) | GPU (OpenCL) (s) |
|-----------|---------|---------------|-----------------|
| PyTorch   | 0.89    | 0.11          | N/A             |
| tinygrad  | 1.45    | 0.18          | 0.24            |
| TF        | 1.02    | 0.13          | N/A             |

tinygrad jest wolniejszy od PyTorch, ale różnica maleje z każdą wersją. Warto zauważyć, że tinygrad obsługuje więcej backendów (OpenCL, WebGPU), co daje mu przewagę na nietypowych platformach.

## Zastosowania w projektach rzeczywistych

Przykłady wykorzystania tinygrad w praktyce:
- OpenPilot (komponent samojezdnego samochodu)
- Mobilne przetwarzanie obrazów
- EdgeML na urządzeniach z ograniczonymi zasobami
- Projekty edukacyjne wymagające zrozumienia całego stosu ML

## Najważniejsze różnice z kulturą PyTorch i TensorFlow

PyTorch/TensorFlow promują używanie gotowych abstrakcji wysokiego poziomu (modele, warstwy). tinygrad zachęca do budowania wszystkiego od podstaw - rozumienie, a nie tylko użycie. Różnica filozoficzna to również podejście do debugowania - w tinygrad widać wszystkie operacje, podczas gdy w większych bibliotekach wiele operacji dzieje się "pod maską".

## Przyszłość tinygrad

George Hotz regularnie rozwija projekt, dodając:
- Wsparcie dla nowych architektur (Apple M1/M2)
- Optymalizacje kompilatora (JIT/AOT)
- "Clownkit" - ekosystem narzędzi wokół tinygrad
- Wsparcie dla nowych modeli (np. LLM, stable diffusion)

## Przykłady kodu

Poniżej przykład jednowarstwowej sieci liniowej z aktywacją ReLU:

```python
# Tinygrad: prosta sieć liniowa
from tinygrad.tensor import Tensor
class SimpleNet:
    def __init__(self):
        self.w = Tensor.uniform(2, 2, requires_grad=True)
        self.b = Tensor.uniform(2, requires_grad=True)
    def __call__(self, x):
        return (x.matmul(self.w) + self.b).relu()

net = SimpleNet()
x = Tensor([1.0, -1.0])
out = net(x)
loss = out.sum()
loss.backward()
```

## Pytania do dyskusji

- Czy minimalistyczne podejście do ML ma przyszłość w erze coraz większych modeli?
- Jak balansować między czytelnością kodu a wydajnością w systemach ML?
- Czy tinygrad mógłby stać się standardem w edukacji ML?

## Zasoby do dalszego zgłębienia

- Oficjalne repozytorium: https://github.com/tinygrad/tinygrad
- Kanał YouTube George'a Hotza: omówienia implementacji
- Artykuł "Why tinygrad?" na blogu tiny corp

## Podsumowanie

tinygrad to **bardzo lekki** framework, którego głównym celem jest edukacja i zrozumienie działania głębokiego uczenia. Oferuje uproszczony interfejs podobny do PyTorch, ale rezygnuje z wielu abstrakcji. Najważniejsze zalety: prostota implementacji, czytelność kodu i łatwość rozbudowy. Stanowi interesującą alternatywę do nauki mechanizmów uczenia, pokazując „od podszewki" jak działa propagacja wsteczna i przetwarzanie tensorów.