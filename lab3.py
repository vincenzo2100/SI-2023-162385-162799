from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# dane wejściowe
rok = np.array([2000, 2002, 2005, 2007, 2010]).reshape((-1, 1))
bezrobocie = np.array([6.5, 7.0, 7.4, 8.2, 9.0])

# tworzenie i trenowanie modelu
model = LinearRegression().fit(rok, bezrobocie)

# wyznaczenie przewidywanych wartości dla lat 2000-2030
lata = np.arange(2000, 2031).reshape((-1, 1))
przewidywane_bezrobocie = model.predict(lata)

# wizualizacja danych
plt.scatter(rok, bezrobocie, color="red")
plt.scatter(lata, przewidywane_bezrobocie)
plt.plot(lata, przewidywane_bezrobocie, color="blue")
for i, txt in enumerate(bezrobocie):
    plt.annotate(
        "%.2f%%" % txt,
        (rok[i], bezrobocie[i]),
        textcoords="offset points",
        xytext=(0, 100),
        ha="center",
        color="red",
    )
    for i, txt in enumerate(przewidywane_bezrobocie):
        plt.annotate(
            "%.3f%%" % txt,
            (lata[i], przewidywane_bezrobocie[i]),
            textcoords="offset points",
            xytext=(0, -20),
            ha="center",
            color="blue",
        )
plt.axhline(12)
plt.xticks(np.arange(2000, 2031, 1))
plt.title("Procent bezrobocia w latach 2000-2030")
plt.xlabel("Rok")
plt.ylabel("Procent bezrobocia")
plt.show()
