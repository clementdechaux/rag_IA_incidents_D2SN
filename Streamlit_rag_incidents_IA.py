import streamlit as st
import os
import typing as t
# LangChain / Langsmith
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import weaviate

from langchain.prompts import ChatPromptTemplate
prompt_generative_context = ChatPromptTemplate.from_template(
"""
En tant qu'expert multidisciplinaire en intelligence artificielle, data science, sociologie et droit, mon objectif est de fournir une analyse complète et nuancée des incidents liés à l'intelligence artificielle. À partir d'un jeu de données spécifique recensant ces incidents, je m'engage à :

Analyser les tendances : Identifier les schémas et tendances émergents dans les incidents d'IA, en utilisant des méthodes de data science avancées pour assurer une compréhension profonde des dynamiques à l'œuvre.

Évaluer l'impact sociétal : Exposer les implications sociologiques de ces incidents, en considérant leur impact sur différentes communautés, pratiques professionnelles, et normes sociales. Cela implique d'explorer les conséquences à court et à long terme de l'intégration de l'IA dans nos vies quotidiennes.

Relever les défis éthiques et légaux : Discuter des questions éthiques et des défis juridiques soulevés par ces incidents, en évaluant comment ils interagissent avec les cadres réglementaires existants et en proposant des voies d'amélioration pour une gouvernance de l'IA plus responsable et équitable.

Fournir des recommandations : Basé sur l'analyse, offrir des recommandations stratégiques pour les décideurs, les développeurs d'IA, et les parties prenantes concernées, visant à prévenir de futurs incidents et à promouvoir une utilisation de l'IA qui soit à la fois éthique et bénéfique pour la société.

Présenter des solutions techniques : Suggérer des solutions techniques innovantes pour adresser les vulnérabilités identifiées dans les systèmes d'IA, en favorisant une conception et un développement d'IA plus sécurisés et transparents.

Je m'engage à réaliser cette analyse avec rigueur et intégrité, en m'appuyant sur mes vastes connaissances et en veillant à fournir des réponses précises, véridiques et contextuellement pertinentes à toutes les questions posées.

Assure-toi de :

Préciser clairement lorsque le contexte ne fournit pas suffisamment d'informations pour élaborer une réponse complète.
Indiquer sans équivoque si une question dépasse le cadre de tes connaissances actuelles.
Voici une question accompagnée de son contexte. 
Analyse soigneusement ces éléments pour fournir une réponse qui prend en compte toutes les informations pertinentes.
Veille à formuler ta réponse de manière concise et précise, en te concentrant sur l'essentiel.

Ajoute une introduction et une conclusion à chacunes de tes réponses.

--- Le contexte:
{context}
--- La question:
{query}
Ta réponse:
"""
    )



def connect_to_weaviate() -> weaviate.client.WeaviateClient:
    client = weaviate.connect_to_wcs(
        cluster_url=os.environ["WEAVIATE_CLUSTER_URL"],
        auth_credentials=weaviate.AuthApiKey(os.environ["WEAVIATE_KEY"]),
        headers={
            "X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"],
        },
    )
    # check that the vector store is up and running
    if client.is_live() & client.is_ready() & client.is_connected():
        print("client is live, ready and connected ")

    assert (
        client.is_live() & client.is_ready()
    ), "Weaviate client is not live or not ready or not connected"
    return client


class Retrieve(object):
    collection_name = "Clement_20240325"

    def __init__(self, query: str, search_params: t.Dict) -> None:
        self.client = connect_to_weaviate()
        assert self.client is not None
        assert self.client.is_live()

        # retrieval
        self.collection = self.client.collections.get(Retrieve.collection_name)
        self.query = query
        self.search_mode = search_params.get("search_mode")
        self.response_count = search_params.get("response_count")

        # output
        self.response = ""
        self.chunk_texts = []
        self.metadata = []

    # retrieve
    def search(self):
        metadata = ["distance", "certainty", "score", "explain_score"]
        if self.search_mode == "hybrid":
            self.response = self.collection.query.hybrid(
                query=self.query,
                # query_properties=["text"],
                limit=self.response_count,
                return_metadata=metadata,
            )
        elif self.search_mode == "near_text":
            self.response = self.collection.query.near_text(
                query=self.query,
                limit=self.response_count,
                return_metadata=metadata,
            )
        elif self.search_mode == "bm25":
            self.response = self.collection.query.bm25(
                query=self.query,
                limit=self.response_count,
                return_metadata=metadata,
            )

    def get_context(self):
        texts = []
        metadata = []
        if len(self.response.objects) > 0:
            for i in range(min([self.response_count, len(self.response.objects)])):
                prop = self.response.objects[i].properties
                texts.append(f"--- \n{prop.get('text')}")
                metadata.append(self.response.objects[i].metadata)
            self.chunk_texts = texts
            self.metadata = metadata

    def close(self):
        self.client.close()

    def process(self):
        self.search()
        self.get_context()
        self.close()


class Generate(object):
    def __init__(self, model: str = "gpt-3.5-turbo-0125", temperature: float = 0.5) -> None:
        self.model = model
        self.temperature = temperature
        llm = ChatOpenAI(model=model, temperature=temperature)

        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt_generative_context,
            output_key="answer",
            verbose=True,
        )

        self.overall_context_chain = SequentialChain(
            chains=[llm_chain],
            input_variables=["context", "query"],
            output_variables=["answer"],
            verbose=True,
        )
        # outputs
        self.answer = ""

    def generate_answer(self, chunk_texts: t.List[str], query: str) -> str:
        response_context = self.overall_context_chain(
            {"context": "\n".join(chunk_texts), "query": query}
        )
        self.answer = response_context["answer"]




##------------La partie Streamlit


st.title('''Politique de l'IA''')
st.header("Comprendre les enjeux de l'IA à travers les incidents")


with st.form("search_form", clear_on_submit=False):
    search_query = st.text_area("Votre question:",
        key="query_input",
        height=20,
        help="""Write a query, a question about your dataset""")

    search_button = st.form_submit_button(label="Ask")


if search_button:
    #  rajouter ici tous le process de la question

    params = {
        "search_mode": "hybrid",
        "response_count": 2,
        "model": "gpt-3.5-turbo-0125",
        "temperature": 0.5,
    }

    ret = Retrieve(search_query, params)
    ret.process()

    gen = Generate()
    gen.generate_answer(ret.chunk_texts, search_query)

    st.write(search_query)
    st.subheader("answer")
    st.write(gen.answer)


    st.subheader("les extraits - contexte")
    st.write(ret.chunk_texts)