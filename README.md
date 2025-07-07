**Personal Website**

This is my portfolio website to present my projects, skills, and experience. I added real time GitHub functionalities and data analytics.

Visit the live site here: https://personalwebsite-oxe6.onrender.com

There are five tabs: home, about me, skills, projects, and contact me

Home
- Personal Intro with name and title
- Links to my github, linkedin, email, and resume

About Me
- My picture
- A short description to get a feel of who I am
- My name, email, and resume at the bottom of the page

Skills
- List of my most notable skills
- Skill bars for languages showing competency levels (estimated)
- Descriptions of my experiences with each skill/technology
- A list of the projects organized under each respective language. This contains both manual entries (private/off-GitHub projects) and live data fetched from my github public portfolio. 

Projects
- Fetches all public github repos and the front page data for each project.
- Generates a graph of the commit history and uses python to create a lowess smoothened curve. 
- Further supplements the graph with timeline statistics and activity metrics.

Contact Me
- An encouragement message at the top to connect to my linkedin.
- Some functionality to send a message to my spreadsheet. 
- Same links as my home page for user convenience.

Note: The github data is fetched every hour to avoid burning through github's fetch token allowance. The graph is updated every time new data is fetched(every hour as well).

index.html: The main website.
server.py: The graph generation. Uses flask to communicate with the frontend.
requirements.txt: A support file for render.
render.yaml: A file telling render how to run the website; needed since there is a backend.
.gitignore: ignores the graphs, the .env, and the pycache when committing.

For secure practices, the github token is stashed in the .env file