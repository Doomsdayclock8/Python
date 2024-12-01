- Dictionary elements don't have index.
- so have to select elements using dict.items()
			```
			key for key,value in dict.items())
- The following snippet stores keys and values inside separate lists by iterating through a dictionary using [[list comprehension]]
```
key_list = ', '.join(key for key,value in dict.items())
value_list = ', '.join(value for key,value in dict.items())
key&value=', '.join(f"{key}={value}" for key,value in dict.items())
```

