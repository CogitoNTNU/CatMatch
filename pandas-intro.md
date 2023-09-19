```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

```


```python
df = pd.read_csv("../data/cats.csv", index_col=0)
```


```python
# Read one column
df["age"]
```


```python
# Read multiple columns
df[["age", "gender"]]


```


```python
# Read one row
df.loc[113]
```


```python
# Read all rows that fulfill a condition
df[df["age"] > 4]
#...
```


```python
# Get the first 10 columns
df.head(10)
```


```python
# Get the last 10 columns
df.tail(10)
```


```python
# Some info about the data
df.info()
```


```python
df.describe()
```


```python
df["age"].value_counts()
```


```python
# Plot some column
df["age"].plot(kind="hist", bins=6)
```


```python
df.plot(kind="scatter", x="height", y="width", xlabel="Height", ylabel="Width")
```


```python
# Create column, this is a silly exampl
df["age_size"] = df["age"] * df["size"]
df
```



```python
# Create a column by labels
def create_label(row):
    if row["age"] > 5:
        return "old"
    else:
        return "young"

df["label"] = df.apply(create_label, axis=1)
df
```


```python
df["label"].value_counts()
```
