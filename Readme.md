# Search Engine for Magic: the Gathering Cards 

This engine enables Natural Language Query over 110,000 cards from the universe of Magic : the Gathering and allows you to search for cards based on description. The Project focuses on fast search algorithms and using Semantic Similarity Based approach to recommend cards based on search Query. 


# How is the Project Structured 

The project consists of FrontEnd and a Backend Service Engine to separate the functionalities. The Project can be setup locally using ```Docker Compose up ```.
- We will be using Streamlit for Frontend for simplicity. 
- Backend will consist of the actual Search Module and an API exposing search Functionality through a ```/search POST``` endpoint.
- The API will be written in ```Flask```


# Explain the Search Algorithm 

In order to make the Search Functionality Performant (across 110,000 cards), we will be using a progressive filtering search algorithm with the following steps : 

0. **Query Expansion** : We will be pre-proccesing the query to find the key-words along with common synonmyms to enrich the user query 
1. **Inverted Indexing Based Search** -> The Idea is to use enriched query tokens to find cards where query words are found to filter the candidates to a smaller amount.
2. **Additional Filter Queries** : While the final version of the project will be using a custom trained model that can take user query and generate MongoDB query parameters, for now we will be using an LLM to generate query Parameters for ideal card. 
3. **BM25** : After retrieving cards from the Database, we will be performing a BM25 scoring on cards to find the top 50 cards. 
4. **Embedding based Similarity Search** : We will be using MiniLM-V2 Embeddings on Query and Card Description Text to find the most similar cards from the 50 cards to generate the final Recommendation. 
5. **Deduplication and Manual Reranking** : This will take care of Different Prints making it in the final list and generate the final List. 