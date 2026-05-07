import numpy as np

def analyze_img(img_array):
    # erstmal schauen ob bild farbe hat oder nicht
    if len(img_array.shape) == 3:
        channel = img_array[:, :, 0]  # einfach ersten channel nehmen
    else:
        channel = img_array

    # pixel binär machen (alles unter 200 = 1)
    binary = (channel < 200).astype(np.uint8)

    # zählen wie viele dunkle pixel pro zeile / spalte
    row_sums = np.sum(binary, axis=1)
    col_sums = np.sum(binary, axis=0)

    # zeile/spalte nur berücksichtigen wenn mindestens 20% der max anzahl hat
    # (runde kanten werden berücksichtigt, ungerade kanten und störpixel nicht)
    row_threshold = np.max(row_sums) * 0.2
    col_threshold = np.max(col_sums) * 0.2

    # indices rausziehen wo genug pixel sind
    core_rows = []

    for i in range(len(row_sums)):
        if row_sums[i] > row_threshold:
            core_rows.append(i)
    
    core_cols = []

    for i in range(len(col_sums)):
        if col_sums[i] > col_threshold:
            core_cols.append(i)

    # grenzen (einfach erstes und letztes nehmen)
    top = core_rows[0]
    bottom = core_rows[-1]
    left = core_cols[0]
    right = core_cols[-1]

    # nur den bereich anschauen (alles außerhalb ignorieren)
    inner_rows = []
    for i in range(top, bottom + 1):
        inner_rows.append(row_sums[i])

    inner_cols = []
    for i in range(left, right + 1):
        inner_cols.append(col_sums[i])
    
    # mittelwert = grobe breite / höhe
    width_px = int(np.round(np.mean(inner_rows)))
    height_px = int(np.round(np.mean(inner_cols)))

    return {
        "width_px": width_px,
        "height_px": height_px
    }


     
if __name__ == "__main__":
    import sys
    from PIL import Image
 
    # Entweder: Bildpfad als Argument (python analyzer.py pfad)
    if len(sys.argv) > 1:
        path = sys.argv[1]
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        print(f"Bild geladen: {path}  ({arr.shape[1]}×{arr.shape[0]} px)")
 
    # Oder: internes Testbild 
    else:
        arr = np.zeros((1123, 794), dtype=np.uint8) * 255
        arr[400:500, 97:697] = 0   # scharzes Rechteck: 600×100 px
        print(f"Testbild: 794×1123 px, Rechteck 600×100 px bei (97,400)")
 
    result = analyze_img(arr)
 
    print()
    print("Ergebnis")
    print(f"Breite:  {result['width_px']}")
    print(f"Höhe:    {result['height_px']}")