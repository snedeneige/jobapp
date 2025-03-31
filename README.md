# jobapp
A job app using ChatGPT and RAG to promote the owner for some position. Mostly for fun and learning.

## To use
git clone https://github.com/snedeneige/jobapp

Make sure to add relevant *.txt files to the *documents* and the *job_descriptions* folders which should be placed in the root folder.

The *documents* folder could contain files like *CV.txt*, *handling_stress.txt* etc, while the files in the *job_descriptions* folder contain info about each job for which there should be an endpoint. For example *netto.txt* would make a /netto endpoint with the text **Why hire me at NETTO**?

Also, add an *agent_instructions.txt* file. These files and folders are all excluded by the .gitignore by default.

## How it works
Each doc in the *documents* folder is transformed into a fixed-length (256 here) vector by OpenAI's *Embedding* API: https://platform.openai.com/docs/guides/embeddings and stored in a DataFrame.

Each user query is likewise transformed and compared to the documents using dot products. If a document scores more than 0.2, it is included as a part of the assistant prompt to OpenAI's Chat completions API before the user query.

For example, the prompt *"Which kinds of experience do you have"* may score high on a CV but lower on a document with hobbies. The chat completion endpoint would then be prompted with the content of the CV file and then the user query.
