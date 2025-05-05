prezentacja 15min

- tinygrad jest jeszcze w becie
- tinygrad celuje w prostote i łatwość w dodawaniu nowych akceleratorów (CUDA, AMD, LLVM).Są firmy, które budują układy obliczeniowe pod AI - groq, tenstorrent. Hardware jest bardzo dobry, lecz software jest słaby. Aby sportować nowy AI czip na Pytorcha, muszą napisać 250 kerneli i wszystkie zopytalizować. Aby sportować do tinygrad, 25 kerneli. tinygrad uzywa bardzo prostego modelu (porównanie do RISC-V)
- API podobne to pytorcha, ale bardziej funkcyjny styl
- Wszystkie operacje są lazy evaluated, pozwala to na lepsze optymalizacje w porówynaniu z Pytorchem (przyklad operations fusing)
- tinycorp, tinybox, decentralizacja
- Pytorch 250 kerneli, tinygrad 25 kerneli (CISC vs RISC)

https://docs.tinygrad.org/
