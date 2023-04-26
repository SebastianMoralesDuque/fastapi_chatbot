# Cargar datos del archivo JSON
with open("intents.json", encoding='utf-8') as archivo:
    datos = json.load(archivo)