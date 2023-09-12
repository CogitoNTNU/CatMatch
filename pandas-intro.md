```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

```


```python
df = pd.read_csv("../data/lidar_data_with_audio_timestamps_oct_28.csv", index_col=0)
```


```python
df = df[["height", "width", "datetime_enter", "datetime_leave"]]
df
```


```python
# Read one column
df["height"]
```


```python
# Read multiple columns
df[["height", "width"]]


```


```python
# Read one row
df.loc[113]
```


```python
# Read all rows that fulfill a condition
df[df["height"] > 200]
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
df["height"].value_counts()
```


```python
# Plot some column
df["height"].plot(kind="hist", bins=6)
```


```python
df.plot(kind="scatter", x="height", y="width", xlabel="Height", ylabel="Width")
```


```python
# Create column
df["area"] = df["height"] * df["width"]
df
```


```python
# Create a column by labels
def create_label(row):
    if row["height"] > 200:
        return "big"
    else:
        return "small"

df["label"] = df.apply(create_label, axis=1)
df
```


```python
df["label"].value_counts()
```
