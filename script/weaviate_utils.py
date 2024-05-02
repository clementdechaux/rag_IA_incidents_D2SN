"""
"""
import os
import weaviate


import weaviate

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
        print(f"client is live, ready and connected ")

    assert (
        client.is_live() & client.is_ready()
    ), "Weaviate client is not live or not ready or not connected"
    return client