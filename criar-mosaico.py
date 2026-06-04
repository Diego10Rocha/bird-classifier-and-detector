from PIL import Image
import os
import math
import random

# Pasta contendo as imagens
pasta = "Detectar passaro\\resultados\\imagens"

# Lista de arquivos válidos
arquivos = [
    f for f in os.listdir(pasta)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    and not f.startswith('b_g')
]

# Número de imagens a exibir
num_imagens = min(9, len(arquivos))

# Seleciona imagens aleatórias sem repetição
arquivos = random.sample(arquivos, num_imagens)

# Configurações do mosaico
colunas = 3
tamanho = (100, 100)

linhas = math.ceil(num_imagens / colunas)

mosaico = Image.new(
    'RGB',
    (colunas * tamanho[0], linhas * tamanho[1]),
    color='white'
)

for i, arquivo in enumerate(arquivos):
    img = Image.open(os.path.join(pasta, arquivo))

    # Redimensiona mantendo proporção e corta excesso
    img.thumbnail(tamanho, Image.Resampling.LANCZOS)

    fundo = Image.new('RGB', tamanho, 'white')
    x = (tamanho[0] - img.width) // 2
    y = (tamanho[1] - img.height) // 2
    fundo.paste(img, (x, y))

    x_grid = (i % colunas) * tamanho[0]
    y_grid = (i // colunas) * tamanho[1]

    mosaico.paste(fundo, (x_grid, y_grid))

mosaico.save("mosaico_aleatorio.png", quality=95)

print(f"Mosaico gerado com {num_imagens} imagens.")