import numpy as np
import os

def generuj_dane_calkowite(folder, ilosc_punktow, wymiary, min_val=0, max_val=100):
    """
    Generuje losowe LICZBY CAŁKOWITE i zapisuje je w formacie: x,y,z;
    """
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    # 1. Dynamiczna nazwa pliku
    nazwa_pliku = f"data_int_{ilosc_punktow}p_{wymiary}d.txt"
    sciezka_pelna = os.path.join(folder, nazwa_pliku)

    # 2. Generowanie macierzy liczb całkowitych
    # np.random.randint(low, high, size)
    dane = np.random.randint(min_val, max_val + 1, size=(ilosc_punktow, wymiary))

    # 3. Zapis do pliku
    with open(sciezka_pelna, 'w') as plik:
        for punkt in dane:
            # Ponieważ to integery, wystarczy str(x) zamiast f"{x:.2f}"
            linia = ",".join([str(x) for x in punkt])
            plik.write(f"{linia};\n")

    print(f"--- GENEROWANIE CAŁKOWITE ZAKOŃCZONE ---")
    print(f"Plik: {nazwa_pliku} (Wartości od {min_val} do {max_val})")
    
    return sciezka_pelna

# --- PARAMETRY ---
pulpit_path = r'D:\Pulpit\PWR\modelowanie i statystyka\data'

# Generujemy np. 100 punktów, 5-wymiarowych, liczby od 1 do 50
sciezka = generuj_dane_calkowite(pulpit_path, 10000, 100, 1, 10)