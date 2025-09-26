import matplotlib
import matplotlib.pyplot as plt
from pyfonts import load_google_font

font = load_google_font("Geist")
matplotlib.rcParams["font.family"] = font.get_name()
matplotlib.rcParams["font.size"] = 14.0

original = [37, 34, 19, 10]
reproduction = [53, 27, 14, 6]

labels = ["1st", "2nd", "3rd", "4th"]
x = range(len(labels))

px = 1 / matplotlib.rcParams["figure.dpi"]

plt.figure(figsize=(906 * px, 810 * px))

plt.bar(
    x,
    original,
    width=0.4,
    label="Litman et al., 2025 (i.e., SPARK 2022)",
    align="center",
    color="#426665",
)
plt.bar(
    [i + 0.4 for i in x],
    reproduction,
    width=0.4,
    label="Reproduction (SPARK 2025)",
    align="center",
    color="#96AAAA",
)

plt.xlabel("Cluster in descending order")
plt.ylabel("Proportion (%)")
plt.title("Litman et al., 2025 (i.e., SPARK 2022) vs. Reproduction (SPARK 2025)")
plt.xticks([i + 0.2 for i in x], labels)
plt.legend()
plt.tight_layout()
plt.show()
