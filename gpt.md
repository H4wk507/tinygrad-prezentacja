# Tinygrad – minimalistyczna biblioteka ML

## Czym jest tinygrad i kto go stworzył

tinygrad to lekka biblioteka do tworzenia i trenowania sieci neuronowych, zaprojektowana z myślą o prostocie. Została stworzona przez George’a “geohot” Hotza – amerykańskiego programistę i hakera, znanego m.in. z jailbreaków i projektu autonomicznych samochodów comma.ai. Biblioteka łączy w sobie prostotę edukacyjnych projektów (micrograd) z inspiracją funkcjonalnością PyTorch. Twórcy podkreślają, że tinygrad obsługuje zarówno trenowanie, jak i inferencję modeli.

* Autor: George Hotz (geohot) – znany z jailbreaków i projektu comma.ai. Od listopada 2022 pracuje nad tinygradem.
* Tinygrad jest utrzymywany przez TinyCorp i ma już \~23 tys. gwiazdek na GitHubie (2025) – szybko rosnące zainteresowanie.
* Hasło: *“something between PyTorch and micrograd”* – łączy prostotę micrograd z interfejsem podobnym do PyTorch.

## Cele i filozofia tinygrad

Główną filozofią tinygrad jest **minimalizm i czytelność kodu**. Biblioteka ma ograniczyć się do niezbędnych elementów, tak by łatwo było ją zrozumieć i rozszerzać. Twórcy podkreślają jej „skrajną prostotę” i łatwość dodawania nowych akceleratorów sprzętowych. Używają humorystycznego stwierdzenia: *“If XLA is CISC, tinygrad is RISC”*, co oznacza, że tinygrad unika nadmiarowych funkcji na rzecz małego zbioru podstawowych operacji.

* **Minimalna liczba operacji:** wewnętrzny IR tinygrada ma tylko 12 typów operacji (tylko dodawanie i mnożenie). Konwolucje i mnożenia macierzy realizowane są przez odpowiednią kombinację tych prostych operacji.
* **Mała baza kodu:** całość biblioteki to niecałe 10 000 linii kodu – dużo mniej niż w dużych frameworkach. To ułatwia naukę i debugowanie.
* **Łatwość rozszerzania:** dzięki prostocie kodu łatwo dodać nowy backend sprzętowy lub optymalizację. Ma to wspierać rozwój autorskiego sprzętu (np. tinybox, przyszłe układy AI).
* **Nastawienie na wydajność:** tinygrad kompiluje specjalny kernel dla każdej operacji, co umożliwia agresywną fuzję operacji i optymalizację kształtów danych. Choć obecnie rzadko jest szybszy niż PyTorch, celem jest osiągnięcie lepszej wydajności (dążenie do 2× przyspieszenia nad PyTorch).

## Architektura biblioteki

Tinygrad opiera się na klasie **Tensor** z wbudowanym autograd (automatyczne różniczkowanie) podobnie jak PyTorch. Wyróżnia go „leniwe” podejście do obliczeń: operacje na tensorach są zapisywane i dopiero przy *realizacji* tensora kompilowane do kodu maszynowego. Dzięki temu wiele operacji można zlać w jeden zoptymalizowany kernel.

* **Autograd i sieci neuronowe:** pełne wsparcie dla propagacji wstecznej (funkcja `backward()` na tensorze). Wbudowana prosta biblioteka `nn` zawiera warstwy (np. liniowe) i optymalizatory (np. `Adam`, `SGD`).
* **Optymalizatory:** standardowe algorytmy typu Adam, SGD itp., z możliwością definiowania learning rate, momentu itd. (moduł `nn.optim`).
* **Leniwe wykonanie (Lazy):** obliczenia są odraczane – np. dodanie tensorów nic nie robi, dopóki nie *zrealizujemy* wyniku. To pozwala scalać (fuzjonować) wiele operacji w jeden kernel GPU.
* **Backendy GPU/CPU:** tinygrad obsługuje wiele akceleratorów: CPU (backend Clang/LLVM) i GPU przez OpenCL, CUDA, Metal (Apple M1 i nowsze) oraz WebGPU/Dawn. Jest też wsparcie dla DSP i GPU mobilnych (flagami `QCOM`, `DSP`, `METAL`). Możliwe jest także równoległe wykorzystanie wielu GPU – funkcja `Tensor.shard` rozdziela tensor na różne urządzenia.
* **Zarządzanie kształtem:** operacje takie jak `reshape`, `permute`, `expand` są traktowane jako „bez kosztu” dzięki specjalnemu śledzeniu kształtu (ShapeTracker). Pozwala to efektywnie przestawiać dane bez kopiowania.
* **JIT (TinyJit):** dostępny prosty JIT – dekorator do przyspieszania powtarzalnych obliczeń. Dekorowana funkcja będzie uruchamiać wcześniej skompilowane kernely zamiast budować graf za każdym razem.
* **Rodzaje danych:** obsługa różnych typów numerycznych (FP32, FP16, BF16 itp.), dzięki czemu można używać np. liczb o niższej precyzji dla wydajności.

## Porównanie do PyTorch i TensorFlow

Tinygrad jest znacznie mniejszy i prostszy od PyTorch czy TensorFlow. Podstawowe operacje (`Tensor`, `add`, `matmul`, autograd) są podobne do tych w PyTorch, ale brakuje wielu udogodnień: np. **brak klasy `nn.Module`** – sieć definiuje się jako zwykłą klasę Pythona z metodą `__call__`. Tinygrad działa leniwie (lazy), podczas gdy PyTorch domyślnie wykonuje operacje od razu (choć i w PyTorch dostępny jest JIT/XLA). Oto najważniejsze różnice:

* **Skala i funkcje:** PyTorch/TF to rozbudowane frameworki liczone w milionach linijek, z ogromnym ekosystemem. Tinygrad to eksperymentalny, \~10k-linijkowy projekt – łatwiejszy do zrozumienia, ale mniej „kompletny” (m.in. brak zaawansowanych API, ekosystemu).
* **API:** Podstawowe API (`Tensor`, operacje matematyczne, autograd) jest podobne, dlatego użytkownicy PyTorch poczują się tu znajomo. Jednak w tinygrad brak jest klas takich jak `nn.Module` czy automatów warstw – warstwy definiuje się ręcznie.
* **Wykonanie:** PyTorch jest „eager” (operacje od razu wykonane), tinygrad robi wszystko leniwie. Powoduje to inne sposoby debugowania: trzeba zrealizować tensor, żeby zobaczyć wynik. Zaletą jest automatyczna fuzja i optymalizacja obliczeń.
* **Optymalizacje:** W PyTorch i TF włożono lata pracy w wydajność (równoległość, tuning GPU, cuDNN itp.). Tinygrad nadrabia to prostotą: każdy kernel jest specjalizowany na dany kształt (shape), co teoretycznie pozwala go bardzo zoptymalizować. W praktyce dziś jest wolniejszy, ale ciągle rozwijany.
* **Cel użycia:** PyTorch/TF stosuje się masowo w przemyśle i badaniach naukowych. Tinygrad raczej do nauki, eksperymentów ze sprzętem i wbudowanych systemów (gdzie liczy się elastyczność).

## Przykłady zastosowań i wydajność

Tinygrad ma realne zastosowania. Jest używany m.in. przez firmę comma.ai do inferencji modelu sterującego w projekcie **openpilot** – na procesorze Snapdragon 845 GPU. Zastąpił tam oryginalny Qualcomm SNPE, działając szybciej i oferując dodatkowe możliwości (np. ładowanie modeli ONNX, mechanizm uwagi). Tinygrad obsługuje pełen cykl forward/backward, więc można na nim trenować sieci – o czym świadczy przykład implementacji prostej sieci liniowej i trenera (przykład w repozytorium).

* **Duże modele:** Tinygrad potrafi uruchomić duże modele generatywne – np. modele LLaMA czy Stable Diffusion. (Trening tak dużych modeli będzie znacznie wolniejszy niż w zoptymalizowanych frameworkach.)
* **Projekty edukacyjne:** Dostępne są tutoriale (np. MNIST), które pokazują, jak łatwo w tinygrad zbudować i przeszkolić sieć. Biblioteka jest na tyle prosta, że można ją samodzielnie przeglądać i eksperymentować z kodem źródłowym.
* **Wydajność:** Obecnie tinygrad nie jest generalnie szybszy od PyTorch/TensorFlow. Jednak ma zalety potencjalnie poprawiające wydajność: kompiluje własne kernela dla każdej operacji oraz stosuje fuzję leniwą. Celem jest osiągnięcie znacznych przyspieszeń w przyszłości (np. dążenie do 2× szybszego niż PyTorch na GPU NVIDIA).
* **Benchmarki sprzętowe:** W testach widziano, że tinygrad na niektórych GPU mobilnych (Snapdragon, Adreno) przewyższał rodzimy framework Qualcomma – co pokazuje, że optymalizacja GPU to silny punkt tinygrad.

## Ostatnie zmiany i nowości (stan na 2025)

W ostatnich latach tinygrad przeszedł znaczące zmiany „pod maską”. W wersji **0.10** (listopad 2024) wprowadzono ogromny refactoring – repo skurczyło się z \~12 000 do \~10 000 linii. Usunięto większość zewnętrznych zależności (np. `numpy`, `pyobjc`), co daje „0 zależności” w Pythonie. Pojawiły się nowe backendy sprzętowe:

* **Mobilne GPU:** flagi `QCOM=1` (Qualcomm Adreno 630) i `DSP=1` (DSP Qualcomm).
* **Specjalizowane akceleratory:** `AMX` (Apple M1 tensor cores) i `XMX` (Intel tensor cores).
* **Chmura:** `CLOUD=1` umożliwia zdalne uruchomienia tinygrad w chmurze.

Wersja 0.10.0 także rozwinęła przejście architektury z **LazyBuffer** na **UOp** (Unified Operation): uproszczono logikę kerneli i rewriterów. Kolejne drobne wydania 0.10.1 i 0.10.2 (luty 2024) przyniosły kolejne usprawnienia: usunięto LazyBuffer, ulepszono harmonogram zadań, dodano pełne wsparcie dla nowego AMD drivera, poprawiono debugowanie i backend WebGPU (Dawn).  Wszystko to zmierza do stabilizacji przed planowaną wersją 1.0.

## Potencjalne zastosowania i dalszy rozwój

Tinygrad dzięki swojej prostocie i elastyczności ma wiele możliwych zastosowań. Może być używany w systemach wbudowanych i na urządzeniach „edge” – wspiera GPU i DSP mobilne, co pozwala na lokalne inferencje bez potężnych serwerów. Prostą implementację można łatwo przenieść na nowe akceleratory sprzętowe – np. FPGA czy dedykowane układy AI (co jest zgodne z planami TinyCorpu, który chce „upowszechnić petaflopa” przez własne chipy).

* **Edge/Embedded ML:** wdrażanie prostych modeli na telefonach, robotach, dronach – wszędzie tam, gdzie potrzebna jest elastyczna biblioteka na ograniczonym sprzęcie.
* **Niestandardowe akceleratory:** tinygrad stworzono z myślą o dodawaniu nowych backendów – jego kod można potraktować jako bazę do własnego frameworka na specyficzne układy ML (np. GPU, FPGA, ASIC).
* **Nauka i prototypowanie:** projekt jest edukacyjnym narzędziem do nauki ML i autograd – studentom pozwala zajrzeć „do środka” frameworka. Łatwo modyfikować kod, co sprzyja eksperymentom naukowym.
* **TinyBox i hardware:** TinyCorp sprzedaje własny komputer TinyBox zoptymalizowany do ML, a w przyszłości planuje rozwijać własne układy AI. Tinygrad ma być fundamentem oprogramowania dla tych systemów.
* **Wspólnota i rozwój:** tinygrad ma aktywną społeczność na GitHubie i Discordzie. Dalszy rozwój będzie polegał na doskonaleniu wydajności, dodawaniu wsparcia dla kolejnych GPU i funkcji ML oraz osiągnięciu stabilności produkcyjnej (wersja 1.0).

Źródła: oficjalne repozytorium tinygrad i dokumentacja, wpisy George’a Hotza oraz materiały TinyCorpu.

https://chatgpt.com/c/68234ec6-ce00-800e-b759-1a9b141efa07
