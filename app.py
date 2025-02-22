from flask import Flask, request, render_template_string
from agent import JobAgent

app = Flask(__name__)

# A simple dictionary to cache agents.
agents = {}

agent_instructions = None

try:
  with open("agent_instructions.txt", "r") as file:
    agent_instructions = file.read()
except FileNotFoundError:
    print("Agent instructions not found.")

def get_job_description(job_id: str) -> str:
    # Open job_descriptions/{job_id}.txt and return the contents.

    try:
        with open(f"job_descriptions/{job_id}.txt", "r") as file:
            return file.read()
    except FileNotFoundError:
        return "Job description not found."

def get_agent(job_id: str) -> JobAgent:
    if job_id not in agents:
        job_description = get_job_description(job_id)
        agents[job_id] = JobAgent(job_description, agent_instructions)
    return agents[job_id]

HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Ask a Question</title>
  <style>
    /* CSS spinner styling */
    #spinner {
      display: none;
      margin: 20px auto;
      border: 8px solid #f3f3f3;
      border-top: 8px solid #3498db;
      border-radius: 50%;
      width: 50px;
      height: 50px;
      animation: spin 1s linear infinite;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
  <h1>Why hire me at {{ company }}?</h1>
  <form id="question-form">
    <textarea id="question" name="question" placeholder="Ask your question here" required style="width:400px; height:100px;"></textarea>
    <button type="submit">Submit</button>
  </form>
  
  <!-- Loading spinner -->
  <div id="spinner"></div>
  
  <div id="response-container">
    <p id="response" style="white-space: pre-wrap;"></p>
  </div>
  <script>
    const form = document.getElementById("question-form");
    const responseParagraph = document.getElementById("response");
    const spinner = document.getElementById("spinner");

    form.addEventListener("submit", async function(event) {
      event.preventDefault(); // Prevent the default form submission.
      const question = document.getElementById("question").value;
      
      // Show the spinner before sending the request.
      spinner.style.display = "block";
      responseParagraph.textContent = "";

      try {
        const res = await fetch(`/{{ job_id }}/ask`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ question: question })
        });
        const data = await res.json();
        responseParagraph.textContent = data.response;
      } catch (error) {
        responseParagraph.textContent = "Error: " + error;
      } finally {
        // Hide the spinner after receiving the response.
        spinner.style.display = "none";
      }
    });
  </script>
</body>
</html>
'''

@app.route("/<job_id>", methods=["GET"])
def index(job_id):
    company_name = job_id[0].upper() + job_id[1:]

    return render_template_string(HTML_TEMPLATE, company=company_name, job_id=job_id)

@app.route("/<job_id>/ask", methods=["POST"])
def ask(job_id):
    data = request.get_json()
    question = data.get("question", "")
    agent = get_agent(job_id)
    response = agent.process_question(question)
    return {"response": response}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)