from langchain.prompts import ChatPromptTemplate


class Prompt(object):
    prompt_generate_groundtruth = ChatPromptTemplate.from_template(
        """
Imagine-toi dans le rôle d'un data scientist et sociologue chevronné,
doté d'une expertise pointue en matière de mobilité quotidienne et des systèmes de transports collectifs au sein de la métropole parisienne.

Respecte les consignes suivantes:
- formule une question simple, de quelques mots
- donne la question et sa réponse au format JSON

--------
Le contexte:
{context}
--------

    "question": "<la question>"
    "reponse": "<la réponse>"

"""
    )

    prompt_generative_context = ChatPromptTemplate.from_template(
        """Imagine-toi dans le rôle d'un data scientist et sociologue chevronné, 
        doté d'une expertise pointue en matière de mobilité quotidienne et des systèmes de transports collectifs au sein de la métropole parisienne.
        En tant qu'intelligence artificielle, mobilise tes vastes connaissances pour aborder les questions qui te sont posées, 
        tout en respectant scrupuleusement la vérité.

Assure-toi de :

Préciser clairement lorsque le contexte ne fournit pas suffisamment d'informations pour élaborer une réponse complète.
Indiquer sans équivoque si une question dépasse le cadre de tes connaissances actuelles.
Voici une question accompagnée de son contexte. 
Analyse soigneusement ces éléments pour fournir une réponse qui prend en compte toutes les informations pertinentes.
Veille à formuler ta réponse de manière concise et précise, en te concentrant sur l'essentiel.

--- Le contexte:
{context}
--- La question:
{query}
Ta réponse:
"""
    )