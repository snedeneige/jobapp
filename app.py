from flask import Flask, render_template, request, render_template_string
from agent import JobAgent
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

logging.info("Starting the Job Application Chatbot...")

agents = {}
agent_instructions = None

try:
  with open("agent_instructions.txt", "r", encoding='utf8') as file:
    agent_instructions = file.read()
except FileNotFoundError as e:
    logging.error("Agent instructions not found: '%s'", e)

def get_job_description(job_id: str) -> str:
    try:
        with open(f"job_descriptions/{job_id}.txt", "r", encoding='utf8') as file:
            return file.read()
    except FileNotFoundError:
        error_message = f"Job description not found for job ID: '{job_id}'"
        logging.error(error_message)
        return error_message

def get_agent(job_id: str) -> JobAgent:
    if job_id not in agents:
        job_description = get_job_description(job_id)
        agents[job_id] = JobAgent(job_id, job_description, agent_instructions)
        logging.info("Created a new agent for job ID: '%s'", job_id)
    return agents[job_id]

@app.route("/<job_id>", methods=["GET"])
def index(job_id):
    company_name = job_id.upper()

    return render_template("index.html", company=company_name, job_id=job_id)

@app.route("/<job_id>/ask", methods=["POST"])
def ask(job_id):
    try:
      data = request.get_json()
      question = data.get("question", "")
      logging.info("Received question: '%s'", question)
      agent = get_agent(job_id)
      response = agent.process_question(question)
      return {"response": response}
    except Exception as e:
      logging.error("Error processing the question: '%s'", e)
      return {"response": "Sorry, perhaps you shouldn't hire me, since I can't process this request."}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
