Like PyG but for tinygrad

## Running tests

```sh
python3 -m pytest test/
```

## Tinygrad suggestions

- boolean indexing for working with bitmasks

```python
t = Tensor([1, 2, 1, 3])
mask = Tensor([True, False, True, True])
split = t[mask] # doesnt work
```