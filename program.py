import numpy as np
import os
import matplotlib.pyplot as plt
import time

nazwa_pliku = 'D:\Pulpit\PWR\modelowanie i statystyka\data\data_int_10000p_100d.txt'

def importuj_dane_z_pliku(sciezka_pliku):
    """
    Importuje dane punktowe z pliku txt. 
    Obsługuje dowolną liczbę wymiarów i automatycznie wykrywa strukturę.
    """
    punkty_lista = []
    
    with open(sciezka_pliku, 'r') as plik:
        for nr_linii, linia in enumerate(plik, 1):
            # Czyszczenie: usuwamy białe znaki i średnik
            czysta_linia = linia.strip().replace(';', '')
            
            if not czysta_linia:
                continue
                
            try:
                # Konwersja linii na listę floatów
                wspolrzedne = [float(x) for x in czysta_linia.split(',')]
                punkty_lista.append(wspolrzedne)
            except ValueError:
                print(f"Pominięto błędną linię nr {nr_linii}: {linia.strip()}")

    # Konwersja na macierz NumPy
    dane = np.array(punkty_lista)
    
    # Automatyczne sprawdzenie wymiarów
    liczba_punktow, wymiary = dane.shape
    print(f"Zaimportowano {liczba_punktow} punktów. Wymiarowość danych: {wymiary}D.")
    
    return dane

def oblicz_odleglosci_od_srodka(macierz_punktow):
    """
    Oblicza odległość euklidesową od środka układu (0,0,...,0) 
    dla danych o dowolnej liczbie wymiarów.
    """
    if macierz_punktow.size == 0:
        return np.array([])

    # axis=1 oznacza, że liczymy normę dla każdego wiersza (punktu) z osobna
    odleglosci = np.linalg.norm(macierz_punktow, axis=1)
    return odleglosci


import numpy as np

def redukacja_wymiarow(macierz_punktow, docelowy_wymiar=1):
    """
    Redukuje wymiarowość danych z uwzględnieniem normalizacji (Z-score).
    """
    if macierz_punktow.shape[1] <= docelowy_wymiar:
        print("Dane mają już wymiar mniejszy lub równy docelowemu. Brak redukcji.")
        return macierz_punktow

    # 1. NORMALIZACJA (Standaryzacja)
    # Obliczamy średnią i odchylenie standardowe dla każdej kolumny
    srednia = np.mean(macierz_punktow, axis=0)
    odchylenie = np.std(macierz_punktow, axis=0)
    
    # Unikamy dzielenia przez zero, jeśli odchylenie wynosi 0
    odchylenie[odchylenie == 0] = 1.0
    
    # Wzór: (x - średnia) / odchylenie
    macierz_znormalizowana = (macierz_punktow - srednia) / odchylenie

    # 2. Obliczanie macierzy kowariancji na ZNORMALIZOWANYCH danych
    kowariancja = np.cov(macierz_znormalizowana, rowvar=False)

    # 3. Obliczanie wartości i wektorów własnych
    wartosci_wlasne, wektory_wlasne = np.linalg.eig(kowariancja)

    # 4. Sortowanie
    idx = np.argsort(wartosci_wlasne)[::-1]
    wartosci_wlasne = wartosci_wlasne[idx]
    wektory_wlasne = wektory_wlasne[:, idx]

    # 5. Wybór wektorów i projekcja
    wybrane_wektory = wektory_wlasne[:, :docelowy_wymiar]
    dane_zredukowane = np.dot(macierz_znormalizowana, wybrane_wektory)
    
    # Dodatkowo: Obliczanie procentu zachowanej wariancji
    wariancja_procent = (wartosci_wlasne[0:docelowy_wymiar].sum() / wartosci_wlasne.sum()) * 100
    
    print(f"Redukcja do {docelowy_wymiar}D zakończona.")
    print(f"Zachowano {wariancja_procent:.2f}% informacji z oryginalnych danych.")
    
    return dane_zredukowane


##petla do wyreksów


try:
    # 1. Przygotowanie danych bazowych
    dane_raw = importuj_dane_z_pliku(nazwa_pliku)
    n_punktow, max_dim = dane_raw.shape
    
    # Tworzymy sufiks do nazw plików, np. "100p_5d"
    sufiks = f"{n_punktow}p_{max_dim}d"
    
    # Referencja (Baseline)
    start_base = time.perf_counter()
    odl_oryginalne = oblicz_odleglosci_od_srodka(dane_raw)
    czas_oryginalny = (time.perf_counter() - start_base) / n_punktow
    
    rozmiar_oryginalny = dane_raw.size # liczba elementów (N * Dim)
    
    # Listy na statystyki
    wymiary_os = []
    bledy_abs = []
    bledy_wzgledne = []
    rozmiary_proc = []
    czasy_proc = []
    bledy_proc = []

    print("\n--- Rozpoczynam głęboką analizę wielokryterialną ---")

    for d in range(max_dim, 0, -1):
        # Redukcja (dla max_dim robimy tylko normalizację, żeby porównanie było fair)
        if d == max_dim:
            # Standaryzacja bez redukcji dla punktu odniesienia 100%
            srednia = np.mean(dane_raw, axis=0)
            odch = np.std(dane_raw, axis=0)
            odch[odch == 0] = 1.0
            zredukowane = (dane_raw - srednia) / odch
        else:
            zredukowane = redukacja_wymiarow(dane_raw, docelowy_wymiar=d)

        # POMIAR CZASU (średni na jeden punkt)
        start = time.perf_counter()
        odl_nowe = oblicz_odleglosci_od_srodka(zredukowane)
        end = time.perf_counter()
        
        czas_nowy = (end - start) / n_punktow
        
        # OBLICZENIA STATYSTYCZNE
        b_abs = np.mean(np.abs(odl_oryginalne - odl_nowe))
        # Błąd względny (unormowany do średniej odległości oryginalnej)
        b_wzgl = b_abs / np.mean(odl_oryginalne)
        
        # Procenty (względem oryginału)
        p_rozmiar = (zredukowane.size / rozmiar_oryginalny) * 100
        p_czas = (czas_nowy / czas_oryginalny) * 100
        p_blad = b_wzgl * 100

        # Zapis do list
        wymiary_os.append(d)
        bledy_abs.append(b_abs)
        bledy_wzgledne.append(b_wzgl)
        rozmiary_proc.append(p_rozmiar)
        czasy_proc.append(p_czas)
        bledy_proc.append(p_blad)

    # --- GENEROWANIE WYKRESÓW ---
    
    # Wykres 1: Błędy
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(wymiary_os, bledy_abs, 'g-o', label='Błąd Absolutny')
    ax1.set_xlabel('Liczba wymiarów')
    ax1.set_ylabel('Błąd Absolutny', color='g')
    ax2 = ax1.twinx()
    ax2.plot(wymiary_os, bledy_wzgledne, 'r--s', label='Błąd Względny')
    ax2.set_ylabel('Błąd Względny', color='r')
    plt.title(f'Analiza Błędu ({n_punktow} próbek, max {max_dim}D)')
    plt.gca().invert_xaxis()
    plt.savefig(f'1_bledy_{sufiks}.png') # <--- Dynamiczna nazwa

    # Wykres 2: Czas
    plt.figure(figsize=(10, 5))
    plt.bar(wymiary_os, czasy_proc, color='orange', alpha=0.7)
    plt.title(f'Czas obliczeń ({n_punktow} próbek, max {max_dim}D) [%]')
    plt.xlabel('Liczba wymiarów')
    plt.ylabel('Procent czasu bazowego')
    plt.savefig(f'2_czas_{sufiks}.png') # <--- Dynamiczna nazwa

    # Wykres 3: Rozmiar vs Błąd
    plt.figure(figsize=(10, 5))
    plt.plot(wymiary_os, rozmiary_proc, 'b-D', label='Rozmiar danych [%]')
    plt.plot(wymiary_os, bledy_proc, 'r-o', label='Błąd wartości [%]')
    plt.fill_between(wymiary_os, bledy_proc, rozmiary_proc, color='gray', alpha=0.2, label='Zysk wydajności')
    plt.title(f'Kompromis Rozmiar vs Błąd ({sufiks})')
    plt.xlabel('Liczba wymiarów')
    plt.ylabel('Wartość procentowa [%]')
    plt.legend()
    plt.gca().invert_xaxis()
    plt.savefig(f'3_rozmiar_vs_blad_{sufiks}.png') # <--- Dynamiczna nazwa

    # Wykres 4: Podsumowanie totalne
    plt.figure(figsize=(10, 6))
    plt.plot(wymiary_os, rozmiary_proc, 'b-', label='Rozmiar [%]')
    plt.plot(wymiary_os, bledy_proc, 'r-', label='Błąd [%]')
    plt.plot(wymiary_os, czasy_proc, 'g-', label='Czas [%]')
    plt.title(f'Podsumowanie Totalne ({sufiks})')
    plt.xlabel('Liczba wymiarów')
    plt.ylabel('Skala procentowa')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    plt.savefig(f'4_podsumowanie_total_{sufiks}.png') # <--- Dynamiczna nazwa

    print(f"\n[Sukces] Wygenerowano wykresy z dopiskiem '_{sufiks}'")
    plt.show()

except Exception as e:
    print(f"Wystąpił błąd podczas analizy: {e}")