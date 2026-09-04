import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier

data = {
    "bill_length_mm": [
        39.1, 39.5, 40.3, 36.7, 39.3,
        46.5, 50.0, 50.8, 48.7, 49.0,
        46.1, 50.0, 45.5, 46.5, 47.2
    ],
    "bill_depth_mm": [
        18.7, 17.4, 18.0, 19.3, 20.6,
        17.9, 16.3, 17.3, 16.0, 16.5,
        13.2, 15.3, 13.7, 14.5, 13.8
    ],
    "flipper_length_mm": [
        181, 186, 195, 193, 190,
        192, 230, 210, 220, 215,
        210, 220, 215, 225, 218
    ],
    "body_mass_g": [
        3750, 3800, 3250, 3450, 3650,
        3500, 5700, 5000, 5200, 5100,
        2900, 4000, 3600, 3900, 3700
    ],
    "species": [
        "Adelie", "Adelie", "Adelie", "Adelie", "Adelie",
        "Chinstrap", "Gentoo", "Gentoo", "Gentoo", "Gentoo",
        "Chinstrap", "Gentoo", "Chinstrap", "Chinstrap", "Chinstrap"
    ]
}

penguin_data = pd.DataFrame(data)

X = penguin_data[
    [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]
]

y = penguin_data["species"]

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X, y)

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model berhasil dibuat!")
print("File model.pkl berhasil disimpan.")