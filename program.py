import numpy as np
import matplotlib.pyplot as plt
import time

nazwa_pliku = r'D:\Pulpit\PWR\modelowanie i statystyka\data\data_int_10000p_100d.txt'

def importuj_dane_z_pliku(sciezka_pliku):
    punkty_lista = []
    with open(sciezka_pliku, 'r') as plik:
        for nr_linii, linia in enumerate(plik, 1):
            czysta_linia = linia.strip().replace(';', '')
            if not czysta_linia: continue
            try:
                wspolrzedne = [float(x) for x in czysta_linia.split(',')]
                punkty_lista.append(wspolrzedne)
            except ValueError:
                print(f"Pominięto błędną linię nr {nr_linii}")
    dane = np.array(punkty_lista)
    print(f"Zaimportowano {dane.shape[0]} punktów. Wymiarowość: {dane.shape[1]}D.")
    return dane

def oblicz_odleglosci_od_srodka(macierz_punktow):
    if macierz_punktow.size == 0: return np.array([])
    return np.linalg.norm(macierz_punktow, axis=1)

def redukacja_wymiarow(macierz_punktow, docelowy_wymiar=1):
    if macierz_punktow.shape[1] <= docelowy_wymiar:
        return macierz_punktow
    srednia = np.mean(macierz_punktow, axis=0)
    odchylenie = np.std(macierz_punktow, axis=0)
    odchylenie[odchylenie == 0] = 1.0
    macierz_znormalizowana = (macierz_punktow - srednia) / odchylenie
    kowariancja = np.cov(macierz_znormalizowana, rowvar=False)
    wartosci_wlasne, wektory_wlasne = np.linalg.eig(kowariancja)
    idx = np.argsort(wartosci_wlasne)[::-1]
    wartosci_wlasne = wartosci_wlasne[idx]
    wektory_wlasne = wektory_wlasne[:, idx]
    wybrane_wektory = wektory_wlasne[:, :docelowy_wymiar]
    dane_zredukowane = np.dot(macierz_znormalizowana, wybrane_wektory)
    wariancja_procent = (wartosci_wlasne[:docelowy_wymiar].sum() / wartosci_wlasne.sum()) * 100
    print(f"Redukcja do {docelowy_wymiar}D (Zachowano {wariancja_procent:.2f}% wariancji)")
    return dane_zredukowane

# --- GŁÓWNA ANALIZA ---
try:
    dane_raw = importuj_dane_z_pliku(nazwa_pliku)
    n_punktow, max_dim = dane_raw.shape
    sufiks = f"{n_punktow}p_{max_dim}d"

    # Rozmiar bazowy
    rozmiar_bazowy_mb = dane_raw.nbytes / (1024 * 1024)

    # Odległości oryginalne
    odl_oryginalne = oblicz_odleglosci_od_srodka(dane_raw)
    srednia_bazowa = np.mean(odl_oryginalne)

    # Czas bazowy – mediana z 5 pomiarów [ms]
    czasy_bazowe = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = np.max(oblicz_odleglosci_od_srodka(dane_raw))
        czasy_bazowe.append((time.perf_counter() - t0) * 1000)
    czas_bazowy_ms = np.median(czasy_bazowe)

    # Listy na statystyki
    wymiary_os      = []
    maks_odl        = []
    srednie_odl     = []
    min_odl         = []
    bledy_abs       = []
    bledy_proc      = []
    bledy_maks_proc = []   # błąd % dla najdalszego punktu
    czasy_ms        = []
    czasy_proc      = []
    rozmiary_mb     = []
    rozmiary_proc   = []

    print("\n--- Rozpoczynam analizę ---")

    for d in range(max_dim, 0, -1):
        if d == max_dim:
            srednia = np.mean(dane_raw, axis=0)
            odch = np.std(dane_raw, axis=0)
            odch[odch == 0] = 1.0
            zredukowane = (dane_raw - srednia) / odch
        else:
            zredukowane = redukacja_wymiarow(dane_raw, docelowy_wymiar=d)

        odl_nowe = oblicz_odleglosci_od_srodka(zredukowane)

        # Czas liczenia maks. odległości – mediana z 5 pomiarów [ms]
        czasy_iter = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = np.max(oblicz_odleglosci_od_srodka(zredukowane))
            czasy_iter.append((time.perf_counter() - t0) * 1000)
        czas_iter_ms = np.median(czasy_iter)

        # Rozmiar
        r_mb = zredukowane.nbytes / (1024 * 1024)

        # Błąd średni
        b_abs  = np.mean(np.abs(odl_oryginalne - odl_nowe))
        b_proc = (b_abs / srednia_bazowa) * 100

        # Błąd dla najdalszego punktu (argmax w zredukowanych)
        idx_maks = np.argmax(odl_nowe)
        blad_punktu_maks = abs(odl_oryginalne[idx_maks] - odl_nowe[idx_maks])
        blad_punktu_maks_proc = (blad_punktu_maks / odl_oryginalne[idx_maks]) * 100

        wymiary_os.append(d)
        maks_odl.append(np.max(odl_nowe))
        srednie_odl.append(np.mean(odl_nowe))
        min_odl.append(np.min(odl_nowe))
        bledy_abs.append(b_abs)
        bledy_proc.append(b_proc)
        bledy_maks_proc.append(blad_punktu_maks_proc)
        czasy_ms.append(czas_iter_ms)
        czasy_proc.append((czas_iter_ms / czas_bazowy_ms) * 100)
        rozmiary_mb.append(r_mb)
        rozmiary_proc.append((r_mb / rozmiar_bazowy_mb) * 100)

    # Różnice maks. odległości względem bazy (max_dim)
    maks_bazowy       = maks_odl[0]
    roznice_maks      = [abs(m - maks_bazowy) for m in maks_odl]
    roznice_maks_proc = [abs((m - maks_bazowy) / maks_bazowy * 100) for m in maks_odl]

    # =========================================================
    # WYKRES 4: ROZKŁAD ODLEGŁOŚCI OD ŚRODKA
    # =========================================================
    plt.figure(figsize=(11, 6))
    plt.plot(wymiary_os, maks_odl,    'r-o', linewidth=2, label='Maks. odległość (najdalszy punkt)')
    plt.plot(wymiary_os, srednie_odl, 'b-s', linewidth=2, label='Średnia odległość')
    plt.plot(wymiary_os, min_odl,     'g-^', linewidth=2, label='Min. odległość (najbliższy punkt)')
    plt.fill_between(wymiary_os, min_odl, maks_odl, alpha=0.08, color='gray', label='Rozpiętość (min–max)')
    plt.fill_between(wymiary_os,
                     [s - np.std(odl_oryginalne) for s in srednie_odl],
                     [s + np.std(odl_oryginalne) for s in srednie_odl],
                     alpha=0.12, color='blue', label='±1 odch. std (poglądowo)')
    plt.title(f'Odległość punktów od środka vs liczba wymiarów ({sufiks})', fontsize=12)
    plt.xlabel('Liczba wymiarów', fontsize=12)
    plt.ylabel('Odległość od środka układu [jednostki znorm.]', fontsize=12)
    plt.legend(fontsize=10, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(f'4_rozklad_odleglosci_{sufiks}.png', dpi=300)
    print(f"Zapisano: 4_rozklad_odleglosci_{sufiks}.png")

    # =========================================================
    # WYKRES 5: RÓŻNICE MAKS. ODLEGŁOŚCI (JEDNOSTKI)
    # =========================================================
    plt.figure(figsize=(11, 6))
    plt.plot(wymiary_os, roznice_maks, 'r-o', linewidth=2, label='|Różnica| maks. odległości vs baza')
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--',
                label=f'Baza ({max_dim}D): {maks_bazowy:.4f}')
    plt.fill_between(wymiary_os, 0, roznice_maks, alpha=0.12, color='red')
    plt.text(0.98, 0.95, f'Wartość bazowa ({max_dim}D) = {maks_bazowy:.4f}',
             transform=plt.gca().transAxes, fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.title(f'Różnica (bezwzgl.) maks. odległości od początku układu ({sufiks})', fontsize=12)
    plt.xlabel('Liczba wymiarów', fontsize=12)
    plt.ylabel('|Różnica| odległości [jednostki znorm.]', fontsize=12)
    plt.legend(fontsize=10, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(f'5_roznice_maks_odl_{sufiks}.png', dpi=300)
    print(f"Zapisano: 5_roznice_maks_odl_{sufiks}.png")

    # =========================================================
    # WYKRES 6: RÓŻNICE MAKS. ODLEGŁOŚCI (PROCENTY)
    # =========================================================
    plt.figure(figsize=(11, 6))
    plt.plot(wymiary_os, roznice_maks_proc, 'r-o', linewidth=2, label='|Różnica| maks. odległości [%]')
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--', label=f'Baza ({max_dim}D) = 0%')
    plt.fill_between(wymiary_os, 0, roznice_maks_proc, alpha=0.12, color='red')
    plt.text(0.98, 0.95, f'Wartość bazowa ({max_dim}D) = {maks_bazowy:.4f}',
             transform=plt.gca().transAxes, fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.title(f'Procentowa różnica (bezwzgl.) maks. odległości od początku układu ({sufiks})', fontsize=12)
    plt.xlabel('Liczba wymiarów', fontsize=12)
    plt.ylabel('|Różnica| [%]', fontsize=12)
    plt.legend(fontsize=10, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(f'6_roznice_maks_odl_proc_{sufiks}.png', dpi=300)
    print(f"Zapisano: 6_roznice_maks_odl_proc_{sufiks}.png")

    # =========================================================
    # WYKRES 7: CZAS OBLICZEŃ [ms]
    # =========================================================
    plt.figure(figsize=(11, 6))
    plt.plot(wymiary_os, czasy_ms, 'b-o', linewidth=2, label='Czas liczenia maks. odległości [ms]')
    plt.axhline(czas_bazowy_ms, color='orange', linewidth=2, linestyle='--',
                label=f'Czas bazowy ({max_dim}D) = {czas_bazowy_ms:.3f} ms')
    plt.fill_between(wymiary_os, 0, czasy_ms, alpha=0.10, color='blue')
    plt.text(0.98, 0.95, f'Wartość bazowa ({max_dim}D) = {czas_bazowy_ms:.3f} ms',
             transform=plt.gca().transAxes, fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.title(f'Czas obliczenia najdalszego punktu od środka ({sufiks})', fontsize=12)
    plt.xlabel('Liczba wymiarów', fontsize=12)
    plt.ylabel('Czas [ms]', fontsize=12)
    plt.legend(fontsize=10, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(f'7_czas_ms_{sufiks}.png', dpi=300)
    print(f"Zapisano: 7_czas_ms_{sufiks}.png")

    # =========================================================
    # WYKRES 8: CZAS OBLICZEŃ [%]
    # =========================================================
    plt.figure(figsize=(11, 6))
    plt.plot(wymiary_os, czasy_proc, 'b-o', linewidth=2, label='Czas [%] względem bazy')
    plt.axhline(100, color='orange', linewidth=2, linestyle='--',
                label=f'Baza ({max_dim}D) = 100% ({czas_bazowy_ms:.3f} ms)')
    plt.fill_between(wymiary_os, 0, czasy_proc, alpha=0.10, color='blue')
    plt.text(0.98, 0.95, f'Wartość bazowa ({max_dim}D) = {czas_bazowy_ms:.3f} ms (100%)',
             transform=plt.gca().transAxes, fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.title(f'Czas obliczenia najdalszego punktu – procentowo ({sufiks})', fontsize=12)
    plt.xlabel('Liczba wymiarów', fontsize=12)
    plt.ylabel('Czas [%]', fontsize=12)
    plt.legend(fontsize=10, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(f'8_czas_proc_{sufiks}.png', dpi=300)
    print(f"Zapisano: 8_czas_proc_{sufiks}.png")

    # =========================================================
    # WYKRES 9: ROZMIAR DANYCH [MB]
    # =========================================================
    plt.figure(figsize=(11, 6))
    plt.plot(wymiary_os, rozmiary_mb, 'g-o', linewidth=2, label='Rozmiar danych [MB]')
    plt.axhline(rozmiar_bazowy_mb, color='purple', linewidth=2, linestyle='--',
                label=f'Rozmiar bazowy ({max_dim}D) = {rozmiar_bazowy_mb:.2f} MB')
    plt.fill_between(wymiary_os, 0, rozmiary_mb, alpha=0.10, color='green')
    plt.text(0.98, 0.95, f'Wartość bazowa ({max_dim}D) = {rozmiar_bazowy_mb:.2f} MB',
             transform=plt.gca().transAxes, fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.title(f'Rozmiar danych po redukcji wymiarów ({sufiks})', fontsize=12)
    plt.xlabel('Liczba wymiarów', fontsize=12)
    plt.ylabel('Rozmiar [MB]', fontsize=12)
    plt.legend(fontsize=10, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(f'9_rozmiar_mb_{sufiks}.png', dpi=300)
    print(f"Zapisano: 9_rozmiar_mb_{sufiks}.png")

    # =========================================================
    # WYKRES 10: ROZMIAR DANYCH [%]
    # =========================================================
    plt.figure(figsize=(11, 6))
    plt.plot(wymiary_os, rozmiary_proc, 'g-o', linewidth=2, label='Rozmiar [%] względem bazy')
    plt.axhline(100, color='purple', linewidth=2, linestyle='--',
                label=f'Baza ({max_dim}D) = 100% ({rozmiar_bazowy_mb:.2f} MB)')
    plt.fill_between(wymiary_os, 0, rozmiary_proc, alpha=0.10, color='green')
    plt.text(0.98, 0.95, f'Wartość bazowa ({max_dim}D) = {rozmiar_bazowy_mb:.2f} MB (100%)',
             transform=plt.gca().transAxes, fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.title(f'Rozmiar danych po redukcji – procentowo ({sufiks})', fontsize=12)
    plt.xlabel('Liczba wymiarów', fontsize=12)
    plt.ylabel('Rozmiar [%]', fontsize=12)
    plt.legend(fontsize=10, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(f'10_rozmiar_proc_{sufiks}.png', dpi=300)
    print(f"Zapisano: 10_rozmiar_proc_{sufiks}.png")

    # =========================================================
    # WYKRES 11: ZESTAWIENIE BŁĄD (najdalszy punkt) + CZAS + PAMIĘĆ [%]
    # =========================================================
    plt.figure(figsize=(11, 6))
    plt.plot(wymiary_os, roznice_maks_proc, 'r-o', linewidth=2,
         label=f'Różnica maks. odległości [%]\n(względem maks. odl. dla {max_dim}D = {maks_bazowy:.4f})')
    plt.plot(wymiary_os, czasy_proc,      'b-s', linewidth=2, label='Czas [%]')
    plt.plot(wymiary_os, rozmiary_proc,   'g-^', linewidth=2, label='Pamięć/Rozmiar [%]')
    plt.axhline(100, color='black', linewidth=1.2, linestyle=':', alpha=0.5,
                label='Poziom bazowy (100%)')
    plt.text(0.02, 0.95,
             f'Baza ({max_dim}D):\n'
             f'  Błąd: |maks_odl(d) - maks_odl({max_dim}D)| / maks_odl({max_dim}D) * 100%\n'
             f'  Maks. odległość bazowa ({max_dim}D) = {maks_bazowy:.4f}\n'
             f'  Czas: {czas_bazowy_ms:.3f} ms\n'
             f'  Pamięć: {rozmiar_bazowy_mb:.2f} MB',
             transform=plt.gca().transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.title(f'Zestawienie: Błąd (najdalszy punkt) / Czas / Pamięć [%] ({sufiks})', fontsize=12)
    plt.xlabel('Liczba wymiarów', fontsize=12)
    plt.ylabel('Wartość procentowa [%]', fontsize=12)
    plt.legend(fontsize=10, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(f'11_zestawienie_proc_{sufiks}.png', dpi=300)
    print(f"Zapisano: 11_zestawienie_proc_{sufiks}.png")

    print(f"\n[Sukces] Wygenerowano wykresy 4–11 z dopiskiem '_{sufiks}'")
    plt.show()

except Exception as e:
    print(f"Błąd: {e}")
    raise