import numpy as np
import matplotlib.pyplot as plt
import time
import os
import csv

# =========================================================
# KONFIGURACJA
# =========================================================
FOLDER_DANYCH   = r'D:\Pulpit\PWR\modelowanie i statystyka\data'
FOLDER_WYKRESY  = r'D:\Pulpit\PWR\modelowanie i statystyka\wykresy_agenta'
FOLDER_WYNIKI   = r'D:\Pulpit\PWR\modelowanie i statystyka\wyniki_agenta'

LICZBY_PROBEK    = [10, 100, 1000, 10000, 100000]
ZAKRESY_WARTOSCI = [(1, 10), (1, 100), (1, 1000), (1, 10000), (1, 100000), (1, 1_000_000)]
WYMIARY          = 100

os.makedirs(FOLDER_DANYCH,  exist_ok=True)
os.makedirs(FOLDER_WYKRESY, exist_ok=True)
os.makedirs(FOLDER_WYNIKI,  exist_ok=True)

# =========================================================
# FUNKCJE POMOCNICZE
# =========================================================
def generuj_dane(folder, n_punktow, wymiary, min_val, max_val):
    nazwa = f"data_int_{n_punktow}p_{wymiary}d_{min_val}-{max_val}.txt"
    sciezka = os.path.join(folder, nazwa)
    if os.path.exists(sciezka):
        print(f"  [PLIK ISTNIEJE] {nazwa} – pomijam generowanie")
        return sciezka
    
    # tworzymy generator
    rng = np.random.default_rng()
    # generowanie liczb całkowitych
    dane = rng.integers(min_val, max_val + 1, size=(n_punktow, wymiary))


    #dane = np.random.randint(min_val, max_val + 1, size=(n_punktow, wymiary))
    
    
    with open(sciezka, 'w') as f:
        for punkt in dane:
            f.write(",".join(str(x) for x in punkt) + ";\n")
    print(f"  [WYGENEROWANO] {nazwa}")
    return sciezka

def wczytaj_dane(sciezka):
    punkty = []
    with open(sciezka, 'r') as f:
        for linia in f:
            czysta = linia.strip().replace(';', '')
            if not czysta: continue
            try:
                punkty.append([float(x) for x in czysta.split(',')])
            except ValueError:
                pass
    return np.array(punkty)

def oblicz_odleglosci(macierz):
    return np.linalg.norm(macierz, axis=1)

# def redukuj_pca(macierz, docelowy_wymiar=1):
#     if macierz.shape[1] <= docelowy_wymiar:
#         return macierz
#     srednia = np.mean(macierz, axis=0)
#     odch    = np.std(macierz,  axis=0)
#     odch[odch == 0] = 1.0
#     znorm   = (macierz - srednia) / odch
#     kowariancja = np.cov(znorm, rowvar=False)
#     wart, wekt  = np.linalg.eig(kowariancja)
#     idx         = np.argsort(wart)[::-1]
#     wybrane     = wekt[:, idx][:, :docelowy_wymiar]
#     return np.dot(znorm, wybrane)

def redukuj_pca(macierz, docelowy_wymiar=1):
    if macierz.shape[1] <= docelowy_wymiar:
        # Zwracamy macierz, puste wartości własne i indeksy (dla spójności)
        return macierz, np.array([1.0]*macierz.shape[1]), np.arange(macierz.shape[1])
        
    srednia = np.mean(macierz, axis=0)
    odch = np.std(macierz, axis=0)
    odch[odch == 0] = 1.0
    znorm = (macierz - srednia) / odch
    
    kowariancja = np.cov(znorm, rowvar=False)
    # Używamy eigh dla stabilności numerycznej macierzy symetrycznych
    wart, wekt = np.linalg.eigh(kowariancja)
    
    idx = np.argsort(wart)[::-1]
    wybrane = wekt[:, idx][:, :docelowy_wymiar]
    zredukowane = np.dot(znorm, wybrane)
    
    return zredukowane, wart, idx


# def zmierz_probe(dane_raw, docelowy_wymiar=1):
#     # Baza: znormalizowane dane w PEŁNYM wymiarze (fair comparison – ta sama skala co PCA)
#     srednia_b = np.mean(dane_raw, axis=0)
#     odch_b    = np.std(dane_raw, axis=0)
#     odch_b[odch_b == 0] = 1.0
#     dane_znorm  = (dane_raw - srednia_b) / odch_b
#     maks_bazowy = np.max(oblicz_odleglosci(dane_znorm))

#     # Redukcja PCA do docelowego wymiaru
#     zredukowane = redukuj_pca(dane_raw, docelowy_wymiar)
#     pamiec_mb   = zredukowane.nbytes / (1024 * 1024)

#     # Czas – mediana 5 pomiarów
#     czasy = []
#     for _ in range(5):
#         t0 = time.perf_counter()
#         _  = np.max(oblicz_odleglosci(zredukowane))
#         czasy.append((time.perf_counter() - t0) * 1000)
#     czas_ms = np.median(czasy)

#     # Błąd: dla punktu najdalszego od środka po redukcji (argmax) – tak jak w programie głównym
#     odl_zred = oblicz_odleglosci(zredukowane)
#     odl_baz  = oblicz_odleglosci(dane_znorm)
#     idx_maks = np.argmax(odl_zred)   # indeks najdalszego punktu po redukcji
#     blad_proc = abs(odl_zred[idx_maks] - odl_baz[idx_maks]) / odl_baz[idx_maks] * 100


#     return czas_ms, pamiec_mb, blad_proc

def zmierz_probe(dane_raw, docelowy_wymiar=1):
    # Redukcja PCA - teraz odbieramy też wartości własne
    zredukowane, wart, idx = redukuj_pca(dane_raw, docelowy_wymiar)
    
    # Pamięć w MB
    pamiec_mb = zredukowane.nbytes / (1024 * 1024)

    # Czas – mediana 5 pomiarów (liczymy czas samej redukcji, to jest ciekawsze badawczo)
    czasy = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = redukuj_pca(dane_raw, docelowy_wymiar) # Mierzymy koszt algorytmu
        czasy.append((time.perf_counter() - t0) * 1000)
    czas_ms = np.median(czasy)

    # --- NOWY SPOSÓB LICZENIA BŁĘDU ---
    # Błąd to stosunek odrzuconych wartości własnych do wszystkich
    calkowita_wariancja = np.sum(wart)
    # Wartości własne są posortowane w idx, bierzemy te, które zostały użyte
    wariancja_zachowana = np.sum(wart[idx][:docelowy_wymiar])
    
    # Błąd: ile % informacji (wariancji) straciliśmy
    if calkowita_wariancja > 0:
        blad_proc = (1 - (wariancja_zachowana / calkowita_wariancja)) * 100
    else:
        blad_proc = 0.0

    return czas_ms, pamiec_mb, blad_proc

# =========================================================
# GŁÓWNA ANALIZA
# =========================================================
wyniki = {}

print("=" * 60)
print("ANALIZA WPŁYWU LICZBY PRÓBEK I ZAKRESU WARTOŚCI")
print("=" * 60)

for (min_v, max_v) in ZAKRESY_WARTOSCI:
    label = f"{min_v}-{max_v}"
    wyniki[label] = {'n': [], 'czas': [], 'pamiec': [], 'blad': []}
    print(f"\n>>> Zakres wartości: [{min_v}, {max_v}]")

    for n in LICZBY_PROBEK:
        print(f"  Próbki: {n:>8}")
        sciezka = generuj_dane(FOLDER_DANYCH, n, WYMIARY, min_v, max_v)
        dane    = wczytaj_dane(sciezka)

        czas_ms, pamiec_mb, blad_proc = zmierz_probe(dane, docelowy_wymiar=1)

        wyniki[label]['n'].append(n)
        wyniki[label]['czas'].append(czas_ms)
        wyniki[label]['pamiec'].append(pamiec_mb)
        wyniki[label]['blad'].append(blad_proc)

        print(f"    Czas: {czas_ms:.4f} ms | Pamięć: {pamiec_mb:.4f} MB | Błąd: {blad_proc:.2f}%")

# =========================================================
# ZAPIS DO CSV
# =========================================================
csv_sciezka = os.path.join(FOLDER_WYNIKI, 'wyniki_agenta.csv')
with open(csv_sciezka, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['zakres', 'n_probek', 'czas_ms', 'pamiec_mb', 'blad_proc'])
    for label, dane_label in wyniki.items():
        for i, n in enumerate(dane_label['n']):
            writer.writerow([
                label, n,
                f"{dane_label['czas'][i]:.6f}",
                f"{dane_label['pamiec'][i]:.6f}",
                f"{dane_label['blad'][i]:.4f}",
            ])
print(f"\n[CSV] Zapisano wyniki: {csv_sciezka}")

# =========================================================
# KOLORY I STYLE – automatycznie dopasowane do liczby zakresów
# =========================================================
etykiety     = list(wyniki.keys())
n_serii      = len(etykiety)
KOLORY       = [plt.cm.tab10(i / max(n_serii - 1, 1)) for i in range(n_serii)]
MARKERY_PULA = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+']
MARKERY      = [MARKERY_PULA[i % len(MARKERY_PULA)] for i in range(n_serii)]

zakresy_x = [f"{a}-{b}" for (a, b) in ZAKRESY_WARTOSCI]
x_pos     = np.arange(len(zakresy_x))
KOLORY_N  = plt.cm.viridis(np.linspace(0, 0.85, len(LICZBY_PROBEK)))

def zapisz_wykres(nazwa):
    sciezka = os.path.join(FOLDER_WYKRESY, nazwa)
    plt.savefig(sciezka, dpi=300, bbox_inches='tight')
    print(f"[WYKRES] Zapisano: {sciezka}")

# =========================================================
# WYKRES A: CZAS [ms] vs LICZBA PRÓBEK
# skala Y automatyczna – bez ylim
# =========================================================
plt.figure(figsize=(11, 6))
for i, label in enumerate(etykiety):
    plt.plot(wyniki[label]['n'], wyniki[label]['czas'],
             color=KOLORY[i], marker=MARKERY[i], linewidth=2,
             label=f'Zakres [{label}]')
plt.xscale('log')
plt.title(f'Czas obliczeń vs Liczba próbek\n(PCA → 1D, wymiary={WYMIARY})', fontsize=12)
plt.xlabel('Liczba próbek (skala log)', fontsize=12)
plt.ylabel('Czas [ms]', fontsize=12)
plt.legend(fontsize=10, shadow=True)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
zapisz_wykres('A_czas_vs_probki.png')

# =========================================================
# WYKRES B: PAMIĘĆ [MB] vs LICZBA PRÓBEK
# skala Y automatyczna – bez ylim
# =========================================================
plt.figure(figsize=(11, 6))
for i, label in enumerate(etykiety):
    plt.plot(wyniki[label]['n'], wyniki[label]['pamiec'],
             color=KOLORY[i], marker=MARKERY[i], linewidth=2,
             label=f'Zakres [{label}]')
plt.xscale('log')
plt.title(f'Zużycie pamięci vs Liczba próbek\n(PCA → 1D, wymiary={WYMIARY})', fontsize=12)
plt.xlabel('Liczba próbek (skala log)', fontsize=12)
plt.ylabel('Pamięć [MB]', fontsize=12)
plt.legend(fontsize=10, shadow=True)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
zapisz_wykres('B_pamiec_vs_probki.png')

# =========================================================
# WYKRES C: BŁĄD [%] vs LICZBA PRÓBEK
# skala Y automatyczna – bez ylim
# =========================================================
plt.figure(figsize=(11, 6))
for i, label in enumerate(etykiety):
    plt.plot(wyniki[label]['n'], wyniki[label]['blad'],
             color=KOLORY[i], marker=MARKERY[i], linewidth=2,
             label=f'Zakres [{label}]')
plt.xscale('log')
plt.title(f'Błąd maks. odległości vs Liczba próbek\n(PCA → 1D, wymiary={WYMIARY})', fontsize=12)
plt.xlabel('Liczba próbek (skala log)', fontsize=12)
plt.ylabel('Błąd [%]', fontsize=12)
plt.legend(fontsize=10, shadow=True)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
zapisz_wykres('C_blad_vs_probki.png')

# =========================================================
# WYKRES D: CZAS [ms] vs ZAKRES WARTOŚCI
# skala Y automatyczna – bez ylim
# =========================================================
plt.figure(figsize=(11, 6))
for j, n in enumerate(LICZBY_PROBEK):
    czasy_dla_n = [wyniki[z]['czas'][j] for z in zakresy_x]
    plt.plot(x_pos, czasy_dla_n,
             color=KOLORY_N[j], marker='o', linewidth=2,
             label=f'n={n}')
plt.xticks(x_pos, zakresy_x, fontsize=10, rotation=15)
plt.title(f'Czas obliczeń vs Zakres wartości\n(PCA → 1D, wymiary={WYMIARY})', fontsize=12)
plt.xlabel('Zakres wartości danych', fontsize=12)
plt.ylabel('Czas [ms]', fontsize=12)
plt.legend(fontsize=10, shadow=True, title='Liczba próbek')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
zapisz_wykres('D_czas_vs_zakres.png')

# =========================================================
# WYKRES E: BŁĄD [%] vs ZAKRES WARTOŚCI
# skala Y automatyczna – bez ylim
# =========================================================
plt.figure(figsize=(11, 6))
for j, n in enumerate(LICZBY_PROBEK):
    bledy_dla_n = [wyniki[z]['blad'][j] for z in zakresy_x]
    plt.plot(x_pos, bledy_dla_n,
             color=KOLORY_N[j], marker='s', linewidth=2,
             label=f'n={n}')
plt.xticks(x_pos, zakresy_x, fontsize=10, rotation=15)
plt.title(f'Błąd maks. odległości vs Zakres wartości\n(PCA → 1D, wymiary={WYMIARY})', fontsize=12)
plt.xlabel('Zakres wartości danych', fontsize=12)
plt.ylabel('Błąd [%]', fontsize=12)
plt.legend(fontsize=10, shadow=True, title='Liczba próbek')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
zapisz_wykres('E_blad_vs_zakres.png')

# =========================================================
# WYKRES F: ZESTAWIENIE CZAS + PAMIĘĆ + BŁĄD
#           normalizowane względem 1. zakresu (= 100%)
#           skala Y automatyczna – wartości mogą przekraczać 100%
# =========================================================
idx_n  = len(LICZBY_PROBEK) - 1
n_maks = LICZBY_PROBEK[-1]

czasy_n   = [wyniki[z]['czas'][idx_n]   for z in zakresy_x]
pamieci_n = [wyniki[z]['pamiec'][idx_n] for z in zakresy_x]
bledy_n   = [wyniki[z]['blad'][idx_n]   for z in zakresy_x]

def norm_bazowa(lst):
    baza = lst[0] if lst[0] != 0 else 1
    return [v / baza * 100 for v in lst]

czasy_norm   = norm_bazowa(czasy_n)
pamieci_norm = norm_bazowa(pamieci_n)
bledy_norm   = norm_bazowa(bledy_n)

plt.figure(figsize=(11, 6))
plt.plot(x_pos, czasy_norm,   'b-o', linewidth=2, label='Czas [% względem 1. zakresu]')
plt.plot(x_pos, pamieci_norm, 'g-s', linewidth=2, label='Pamięć [% względem 1. zakresu]')
plt.plot(x_pos, bledy_norm,   'r-^', linewidth=2, label='Błąd [% względem 1. zakresu]')
plt.axhline(100, color='black', linewidth=1.2, linestyle=':', alpha=0.5,
            label='Poziom bazowy (100% = 1. zakres)')
plt.xticks(x_pos, zakresy_x, fontsize=10, rotation=15)
# Brak plt.ylim – skala automatyczna, nie obcina wartości > 100%
plt.title(f'Zestawienie: Czas / Pamięć / Błąd vs Zakres wartości\n'
          f'(względem 1. zakresu = 100%, n={n_maks}, wymiary={WYMIARY})', fontsize=12)
plt.xlabel('Zakres wartości danych', fontsize=12)
plt.ylabel('Wartość względna [%]', fontsize=12)
plt.legend(fontsize=10, shadow=True)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
zapisz_wykres('F_zestawienie_vs_zakres.png')

print("\n[SUKCES] Wygenerowano wykresy A–F oraz plik CSV z wynikami.")
plt.show()