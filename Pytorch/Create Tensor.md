### Conversion from other Formats:

|Format|Steps Required Before Conversion|
|---|---|
|**DataFrame**|Extract values using `df.values` or `df.to_numpy()`|
|**CSV**|Load with `pandas.read_csv()` or `numpy.loadtxt()`|
|**JSON**|Flatten or preprocess to a numeric structure|
|**NumPy**|Direct conversion using `torch.from_numpy()`|
|**Lists**|Direct conversion using `torch.tensor()`|
