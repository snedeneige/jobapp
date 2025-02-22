# jobapp
A job app using ChatGPT and RAG to promote the owner for some position.

## To use
git clone https://github.com/snedeneige/jobapp

Make sure to add relevant *.txt files to the *documents* and the *job_descriptions* folders.

Also, add an *agent_instructions.txt* file. These are all excluded by the .gitignore by default.

## How it works
Each query is transformed into a vector by OpenAI's *Embedding* API: https://platform.openai.com/docs/guides/embeddings.
