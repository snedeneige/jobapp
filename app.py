from flask import Flask, render_template, request
from agent import JobAgent
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

logging.info("Starting the Job Application Chatbot...")

_agents : dict[str, JobAgent] = {}
_agent_instructions : str | None = None

try:
    with open("agent_instructions.txt", "r", encoding='utf8') as file:
        _agent_instructions = file.read()
except FileNotFoundError as e:
    logging.error("Agent instructions not found: '%s'", e)

def read_job_description(job_id: str) -> str:
    try:
        with open(f"job_descriptions/{job_id}.txt", "r", encoding='utf8') as file:
            return file.read()
    except FileNotFoundError:
        error_message = f"Job description not found for job ID: '{job_id}'"
        logging.error(error_message)
        return error_message

def get_agent(job_id: str) -> JobAgent:
    if job_id not in _agents:
      job_description = read_job_description(job_id)
      _agents[job_id] = JobAgent(job_id, job_description, _agent_instructions)
      logging.info("Created a new agent for job ID: '%s'", job_id)
    return _agents[job_id]

@app.route("/<job_id>", methods=["GET"])
def index(job_id):
    return render_template("index.html", company=job_id.upper(), job_id=job_id)

@app.route("/<job_id>/ask", methods=["POST"])
def ask(job_id):
    try:
        data = request.get_json()
        question = data.get("question", "")
        logging.info("Received question: '%s'", question)
        agent = get_agent(job_id)
        response = agent.process_question(question)
        logging.info("Returning response: '%s'", response)
        return {"response": response}
    except Exception as e:
        logging.error("Error processing the question: '%s'", e)
        return {"response": "Sorry, perhaps you shouldn't hire me, since I can't process this request."}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
