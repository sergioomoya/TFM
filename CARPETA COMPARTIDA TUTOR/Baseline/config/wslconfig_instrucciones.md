# Limitar RAM de WSL (evitar OOM en Cursor)

Si `vmmemWSL` consume demasiada RAM y Cursor se cierra por OOM:

1. Crea o edita `C:\Users\<TU_USUARIO>\.wslconfig`

2. Añade (ajusta según tu RAM total):

```
[wsl2]
memory=6GB
swap=2GB
processors=4
```

3. Cierra WSL y reinicia:
   ```
   wsl --shutdown
   ```
   (en PowerShell como admin o desde CMD)

4. Abre de nuevo tu terminal WSL o reinicia el PC.

| RAM total | memory sugerido |
|-----------|-----------------|
| 8 GB      | 2-3 GB          |
| 16 GB     | 4-6 GB          |
| 32 GB     | 8-12 GB         |
