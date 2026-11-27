# Data

The two corrosion datasets used in this study were published by:

> Hakimian, S., Pourrahimi, S., Bouzid, A.-H., Hof, L.A., 2023. Application of machine
> learning for the classification of corrosion behavior in different environments for
> material selection of stainless steels. *Computational Materials Science* 228, 112352.
> https://doi.org/10.1016/j.commatsci.2023.112352

Obtain the files from the original publication and place them here with these names:

| File | Description |
|---|---|
| `Dataset No.1.csv` | 4,019 samples; environments given as electrolyte type, concentration and temperature |
| `Dataset No.2.csv` | 1,544 samples at 20 °C; environments given as [Cl⁻], [H⁺], [OH⁻] and [S²⁻] |

The `.gitignore` excludes `data/*.csv`. If the original licence allows redistribution,
remove that line and commit the files.
