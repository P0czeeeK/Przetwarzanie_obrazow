import cv2
import matplotlib.pyplot as plt
import numpy as np
import urllib.request

url = "https://img.magnific.com/darmowe-zdjecie/8-bitowa-scena-pikseli-graficznych-z-natura_23-2151120963.jpg?semt=ais_hybrid&w=740&q=80"

req_obj = urllib.request.Request(
    url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
)

with urllib.request.urlopen(req_obj) as req:
  arr = np.asarray(bytearray(req.read()), dtype=np.uint8)

img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Zdalny obraz źródłowy")
plt.axis("off")

plt.subplot(1, 2, 2)
colors = ("r", "g", "b")
channel_names = ("Czerwony (R)", "Zielony (G)", "Niebieski (B)")

for i, col in enumerate(colors):
  hist = cv2.calcHist([img], [i], None, [256], [0, 256])
  plt.plot(hist, color=col, label=channel_names[i], alpha=0.8)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist_full = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.plot(hist_full, color="black", linestyle="--", label="Całość (Gray)", linewidth=1.5)

plt.title("Histogram dla kanałów i całego obrazu")
plt.xlabel("Wartość intensywności piksela (0 - 255)")
plt.ylabel("Liczba pikseli")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.show()

def evaluate_image_quality(gray_img):
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()
    total_pixels = gray_img.size

    dark_pixels = np.sum(hist[:10]) / total_pixels
    bright_pixels = np.sum(hist[245:]) / total_pixels

    std_dev = np.std(gray_img)
    mean_val = np.mean(gray_img)

    significant_bins = np.where(hist > (total_pixels * 0.001))[0]
    dynamic_range = (
        significant_bins[-1] - significant_bins[0]
        if len(significant_bins) > 0
        else 0
    )

    print("\n" + "=" * 50)
    print(" RAPORT Z OCENY JAKOŚCI OBRAZU (ALGORYTM HISTOGRAMOWY) ")
    print("=" * 50)
    print(f"• Średnia jasność obrazu: {mean_val:.2f} / 255")
    print(f"• Kontrast (odchylenie standardowe): {std_dev:.2f}")
    print(f"• Szerokość zakresu dynamicznego: {dynamic_range} / 255")
    print(f"• Udział pikseli niedoświetlonych (cienie): {dark_pixels * 100:.2f}%")
    print(
        f"• Udział pikseli prześwietlonych (światła): {bright_pixels * 100:.2f}%"
    )

    quality_score = "Dobra"
    comments = []

    if std_dev < 30:
        quality_score = "Niska (Obraz niskokontrastowy / 'płaski')"
        comments.append(
            "Wąski rozkład wartości w histogramie – mała rozpiętość tonalna."
        )
    elif dark_pixels > 0.15:
        quality_score = "Średnia / Niedoświetlona"
        comments.append(
            "Duża kumulacja pikseli w lewej stronie histogramu (niedoświetlenie/utrata detali w cieniach)."
        )
    elif bright_pixels > 0.15:
        quality_score = "Średnia / Prześwietlona"
        comments.append(
            "Duża kumulacja pikseli w prawej stronie histogramu (przepalenia/prześwietlenia)."
        )
    else:
        comments.append(
            "Histogram jest dobrze rozłożony w całym zakresie, co świadczy o prawidłowym naświetleniu i kontraście."
        )

    print(f"\n=> SZACOWANA JAKOŚĆ: {quality_score}")
    print("Uwagi analityczne:")
    for comment in comments:
        print(f"  - {comment}")
    print("=" * 50)


evaluate_image_quality(gray)