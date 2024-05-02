import pandas as pd
import uuid

# Fonction pour découper le texte en chunks avec chevauchement
def chunkit(input_: list, window_size: int = 3, overlap: int = 1) -> list:
    assert overlap < window_size, f"overlap {overlap} needs to be smaller than window size {window_size}"
    start_ = 0
    chunks = []
    while start_ + window_size <= len(input_):
        chunks.append(input_[start_: start_ + window_size])
        start_ += window_size - overlap
    if start_ < len(input_):
        chunks.append(input_[start_:])
    return ["\n".join(chunk) for chunk in chunks]

if __name__ == "__main__":
    # Charger le fichier Excel
    file_path = 'data/sources/wikipedia_clement/AIAAIC Repository.xlsx'
    xl = pd.ExcelFile(file_path)
    df = xl.parse(xl.sheet_names[0])  # Charger la première feuille

    # Combinaison des colonnes d'intérêt en une chaîne de texte par ligne
    columns_of_interest = ['Headline/title', 'Type', 'Released', 'Occurred', 'Country(ies)', 
                           'Sector(s)', 'Deployer(s)', 'Developer(s)', 'System name(s)', 
                           'Technology(ies)', 'Purpose(s)', 'Media trigger(s)', 'Issue(s)', 
                           'Transparency', 'Description/links']
    combined_text = df[columns_of_interest].fillna('').apply(lambda row: ' | '.join(row.values.astype(str)), axis=1)
    
    # Nettoyer et diviser le texte combiné en lignes
    cleaned_text = combined_text.str.strip().tolist()

    # Appliquer le traitement de chunk sur le texte combiné
    chunked_text = chunkit(cleaned_text, window_size=5, overlap=2)

    # Créer des données fictives pour les colonnes supplémentaires
    dates_creation = ['2024-03-25' for _ in range(len(chunked_text))]
    auteurs = ['Auteur Exemple' for _ in range(len(chunked_text))]
    titres = ['Titre Exemple' for _ in range(len(chunked_text))]
    token_counts = [len(txt.split()) for txt in chunked_text]

    # Préparer les données pour la sauvegarde
    data = pd.DataFrame({
        'text': chunked_text,
        'uuid': [str(uuid.uuid4()) for _ in range(len(chunked_text))],
        'token_count': token_counts,
    })

    # Sauvegarder au format JSON
    output_file_json = '/Users/clement/Documents/data project/rag - IA incidents/data/rag/incident_IA.json'
    data.to_json(output_file_json, force_ascii=False, orient="records", indent=4)

    print(f"-- saved to {output_file_json}")


