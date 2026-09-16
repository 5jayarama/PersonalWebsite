import os
import json
import time
import threading
import resend
from datetime import datetime, timedelta, timezone
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS

load_dotenv()

GITHUB_USERNAME = '5jayarama'
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

resend.api_key = os.getenv('RESEND_API_KEY')
CONTACT_RECIPIENT = os.getenv('CONTACT_RECIPIENT')  # your email, where form submissions go
CONTACT_FROM = os.getenv('CONTACT_FROM', 'onboarding@resend.dev')  # sender address (see note below)

if os.environ.get('RENDER'):
    REPO_DATA_FOLDER = '/opt/render/project/src/static/repo_data'
else:
    REPO_DATA_FOLDER = os.path.join('static', 'repo_data')

os.makedirs(REPO_DATA_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

GITHUB_HEADERS = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json',
    'User-Agent': f'PersonalWebsite-{GITHUB_USERNAME}'
}

def fetch_repositories():
    """Fetch all repositories for the user"""
    url = f"https://api.github.com/users/{GITHUB_USERNAME}/repos"
    params = {'sort': 'updated', 'per_page': 100}

    response = requests.get(url, params=params, headers=GITHUB_HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching repositories: {response.status_code}")
        if response.status_code == 403:
            print(f"Rate limit info: {response.headers.get('X-RateLimit-Remaining', 'unknown')} requests remaining")
        return []

def fetch_commits(repo_name, per_page=100):
    """Fetch commits for a specific repository"""
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/commits"
    params = {'per_page': per_page}

    print(f"Fetching commits for {repo_name} from: {url}")
    response = requests.get(url, params=params, headers=GITHUB_HEADERS)

    if response.status_code == 200:
        commits = response.json()
        print(f"Successfully fetched {len(commits)} commits for {repo_name}")
        return [commit['commit']['author']['date'] for commit in commits]
    else:
        print(f"Error fetching commits for {repo_name}: HTTP {response.status_code}")
        if response.status_code == 403:
            print(f"Rate limit info: {response.headers.get('X-RateLimit-Remaining', 'unknown')} requests remaining")
            reset_time = response.headers.get('X-RateLimit-Reset')
            if reset_time:
                reset_datetime = datetime.fromtimestamp(int(reset_time))
                print(f"Rate limit resets at: {reset_datetime}")
        elif response.status_code == 404:
            print(f"Repository {repo_name} not found or no access")
        elif response.status_code == 409:
            print(f"Repository {repo_name} is empty (no commits)")
        else:
            print(f"Unexpected error for {repo_name}: {response.text[:200]}")
        return []

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300  # Cache static files for 5 minutes
app.config['JSON_SORT_KEYS'] = False  # Preserve JSON key order

def format_date(date_obj):
    """Format a date/datetime as M/D/YY without a leading zero on month/day."""
    try:
        return date_obj.strftime('%-m/%-d/%y')
    except ValueError:
        try:
            return date_obj.strftime('%#m/%#d/%y')
        except ValueError:
            formatted = date_obj.strftime('%m/%d/%y')
            return formatted.lstrip('0').replace('/0', '/')

def compute_commit_stats(commit_dates):
    """Summarize commit activity for a repository (no graph/image involved).

    Returns a dict of stats, or None if there are no commits.
    """
    if not commit_dates:
        return None

    dates = [datetime.fromisoformat(date.replace('Z', '+00:00')) for date in commit_dates]
    dates.sort()

    first_date = dates[0].date()
    last_commit_date = dates[-1].date()
    today = datetime.now(timezone.utc).date()

    timeline_days = (today - first_date).days + 1
    active_day_set = {d.date() for d in dates}
    active_days = len(active_day_set)
    total_commits = len(dates)
    avg_commits_per_day = round(total_commits / active_days, 1) if active_days > 0 else 0

    start_date_display = format_date(datetime.combine(first_date, datetime.min.time()))
    last_commit_display = format_date(datetime.combine(last_commit_date, datetime.min.time()))
    today_display = format_date(datetime.combine(today, datetime.min.time()))

    return {
        'total_commits': total_commits,
        'active_days': active_days,
        'avg_commits_per_day': avg_commits_per_day,
        'timeline_days': timeline_days,
        'date_range': f"{start_date_display} to {today_display}",
        'last_commit_date': last_commit_display
    }

def fetch_readme(repo_name):
    """Fetch a repository's README as raw markdown/text (or None if missing)."""
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/readme"
    headers = {**GITHUB_HEADERS, 'Accept': 'application/vnd.github.raw'}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 404:
            print(f"No README found for {repo_name}")
            return None
        else:
            print(f"Error fetching README for {repo_name}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception fetching README for {repo_name}: {e}")
        return None

def fetch_repo_data(repo_name):
    """Fetch commit stats and README for a repository (no graph/image involved)."""
    commit_dates = fetch_commits(repo_name)
    stats = compute_commit_stats(commit_dates)
    readme_content = fetch_readme(repo_name)
    return {
        'stats': stats,
        'readme': readme_content
    }

def cache_repo_data(repo_name, save_path):
    """Fetch a repository's stats + README and cache them to a JSON file."""
    print(f"Fetching data for {repo_name}...")
    data = fetch_repo_data(repo_name)

    if data['stats'] is None and data['readme'] is None:
        print(f"No data found for {repo_name}")
        return None

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Data cached for {repo_name}")
    return data

def generate_all_repo_data():
    """Fetch and cache stats + README for ALL repositories"""
    print("Fetching data for ALL repositories...")

    try:
        repos = fetch_repositories()
        total_repos = len(repos)
        successful = 0
        failed = 0

        print(f"Found {total_repos} repositories to process")

        for i, repo in enumerate(repos, 1):
            repo_name = repo['name']
            data_path = os.path.join(REPO_DATA_FOLDER, f"{repo_name}_data.json")

            print(f"[{i}/{total_repos}] Fetching data for {repo_name}...")

            try:
                data = cache_repo_data(repo_name, data_path)

                if data is not None:
                    print(f"[{i}/{total_repos}] Cached data for {repo_name}")
                    successful += 1
                else:
                    print(f"[{i}/{total_repos}] No data found for {repo_name}")
                    failed += 1

            except Exception as e:
                print(f"[{i}/{total_repos}] Error fetching {repo_name}: {e}")
                failed += 1

            # Small delay between requests to be nice to GitHub API
            if i < total_repos:
                time.sleep(0.1)  # 100ms between repos

        print(f"Data fetch complete: {successful} successful, {failed} failed out of {total_repos} repositories")
        return successful

    except Exception as e:
        print(f"Error in generate_all_repo_data: {e}")
        return 0

def hourly_repo_data_refresh():
    print("Starting background repo data refresh system...")
    print(f"Server timezone: {datetime.now()}")
    print(f"UTC time: {datetime.now(timezone.utc)}")

    while True:
        try:
            print(f"Waiting 1 hour for next refresh... (next update at {(datetime.now() + timedelta(hours=1)).strftime('%H:%M:%S')})")
            time.sleep(3600)

            print("HOURLY REFRESH: Refreshing all repository data...")
            print(f"Current time: {datetime.now()}")
            successful = generate_all_repo_data()

            if successful > 0:
                print(f"Hourly refresh completed successfully ({successful} repos updated)")
            else:
                print("Hourly refresh completed but no repo data was generated")

        except Exception as e:
            print(f"Error in hourly refresh cycle: {e}")
            # Continue the loop even if there's an error
            time.sleep(60)  # Wait 1 minute before retrying

def start_background_repo_data_system():
    """Start the background repo data refresh system"""
    refresh_thread = threading.Thread(target=hourly_repo_data_refresh, daemon=True)
    refresh_thread.start()
    print("Background repo data system started - refreshing all repo data every hour")

# FLASK ROUTES
@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_file('index.html')

@app.route('/styles.css')
def serve_styles():
    return send_file('styles.css')

@app.route('/api/repositories')
def get_repositories():
    """Get list of repositories with preloaded stats + README info"""
    try:
        repos = fetch_repositories()
        total_repos = len(repos)

        repo_data = []
        for repo in repos:
            repo_name = repo['name']
            data_filename = f"{repo_name}_data.json"
            data_path = os.path.join(REPO_DATA_FOLDER, data_filename)

            details = None
            if os.path.exists(data_path):
                try:
                    # Check file age (for display purposes)
                    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(data_path))
                    age_minutes = int(file_age.total_seconds() / 60)

                    with open(data_path, 'r', encoding='utf-8') as f:
                        cached = json.load(f)

                    details = {
                        'stats': cached.get('stats'),
                        'readme': cached.get('readme'),
                        'ready': True,
                        'age_minutes': age_minutes
                    }
                    print(f"Preloaded data available for {repo_name} (age: {age_minutes}m)")

                except Exception as e:
                    print(f"Error loading cached data for {repo_name}: {e}")
                    details = {'ready': False, 'reason': 'corrupted'}
            else:
                print(f"No preloaded data for {repo_name}")
                details = {'ready': False, 'reason': 'missing'}

            repo_data.append({
                'name': repo['name'],
                'description': repo['description'],
                'stars': repo['stargazers_count'],
                'forks': repo['forks_count'],
                'language': repo['language'],
                'updated_at': repo['updated_at'],
                'html_url': repo['html_url'],
                'details': details
            })

        ready_count = sum(1 for repo in repo_data if repo['details']['ready'])
        print(f"Repository API called: {ready_count}/{total_repos} repo data ready")

        return jsonify({
            "status": "success",
            "repositories": repo_data,
            "data_ready": ready_count,
            "total_repos": len(repo_data)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/contact', methods=['POST'])
def contact():
    """Send a contact form submission as an email"""
    data = request.get_json(silent=True) or request.form

    name = (data.get('name') or '').strip()
    email = (data.get('email') or '').strip()
    message = (data.get('message') or '').strip()

    if not name or not email or not message:
        return jsonify({"status": "error", "message": "All fields are required."}), 400

    if not resend.api_key or not CONTACT_RECIPIENT:
        print("Contact form error: Resend API key or recipient not configured")
        return jsonify({"status": "error", "message": "Contact form is not configured."}), 500

    try:
        resend.Emails.send({
            "from": CONTACT_FROM,
            "to": [CONTACT_RECIPIENT],
            "reply_to": email,
            "subject": f"Portfolio contact form: {name}",
            "text": f"From: {name} <{email}>\n\n{message}"
        })
    except Exception as e:
        print(f"Contact form send failed: {e}")
        return jsonify({"status": "error", "message": "Failed to send message."}), 500

    try:
        resend.Emails.send({
            "from": CONTACT_FROM,
            "to": [email],
            "subject": "Thanks for reaching out!",
            "text": (
                f"Hi {name},\n\n"
                "Thank you so much for reaching out! This is an automated message to confirm that I have received your message in my email inbox and will get back to you soon.\n\n"
                "For your records, here's what you sent:\n"
                f"\"{message}\"\n\n"
                "Best,\nAdarsh Jayaram"
            )
        })
    except Exception as e:
        # Don't fail the whole request if just the auto-reply fails —
        # the notification to you already went through.
        print(f"Auto-reply send failed: {e}")

    return jsonify({"status": "success", "message": "Message sent."}), 200

@app.route('/api/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "repo_data_folder": REPO_DATA_FOLDER
    })


print("Starting GitHub Projects Server...")
# PRE-BUILD: Fetch all repo data (stats + READMEs) FIRST
print("Pre-fetching all repository data before starting server...")
successful = generate_all_repo_data()
print(f"Pre-fetch complete: {successful} repos processed")
# Start background system BEFORE app.run()
start_background_repo_data_system()
if __name__ == '__main__':
    # Start Flask app (this blocks, so put it last)
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    print(f"Server starting on port {port}")
    app.run(debug=debug, host='0.0.0.0', port=port)