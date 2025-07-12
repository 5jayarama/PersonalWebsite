from flask import Flask, render_template, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
from dateutil.relativedelta import relativedelta
import calendar
import math
import threading
import time
from flask import send_file
from dotenv import load_dotenv
import matplotlib
from matplotlib.ticker import FuncFormatter
matplotlib.use('Agg')  # Use non-GUI backend for server environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Use Render disk mount point for persistent storage
if os.environ.get('RENDER'):
    GRAPHS_FOLDER = '/opt/render/project/src/static/graphs'
else:
    GRAPHS_FOLDER = os.path.join('static', 'graphs')

# Ensure the directory exists
os.makedirs(GRAPHS_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)  # Enable CORS for web requests

# Configure paths
STATIC_FOLDER = 'static'
GITHUB_USERNAME = '5jayarama'
load_dotenv()
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

# GitHub API headers with authentication
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
    
    print(f"🔍 Fetching commits for {repo_name} from: {url}")
    response = requests.get(url, params=params, headers=GITHUB_HEADERS)
    
    if response.status_code == 200:
        commits = response.json()
        print(f"✅ Successfully fetched {len(commits)} commits for {repo_name}")
        return [commit['commit']['author']['date'] for commit in commits]
    else:
        print(f"❌ Error fetching commits for {repo_name}: HTTP {response.status_code}")
        if response.status_code == 403:
            print(f"⚠️ Rate limit info: {response.headers.get('X-RateLimit-Remaining', 'unknown')} requests remaining")
            reset_time = response.headers.get('X-RateLimit-Reset')
            if reset_time:
                reset_datetime = datetime.fromtimestamp(int(reset_time))
                print(f"⏰ Rate limit resets at: {reset_datetime}")
        elif response.status_code == 404:
            print(f"🚫 Repository {repo_name} not found or no access")
        elif response.status_code == 409:
            print(f"📭 Repository {repo_name} is empty (no commits)")
        else:
            print(f"🔥 Unexpected error for {repo_name}: {response.text[:200]}")
        return []

# Added this new function to check rate limits
def check_rate_limit():
    """Check GitHub API rate limit status"""
    url = "https://api.github.com/rate_limit"
    response = requests.get(url, headers=GITHUB_HEADERS)
    if response.status_code == 200:
        data = response.json()
        core = data['resources']['core']
        print(f"🔄 Rate limit: {core['remaining']}/{core['limit']} requests remaining")
        reset_time = datetime.fromtimestamp(core['reset'])
        print(f"🕐 Resets at: {reset_time.strftime('%H:%M:%S')}")
        return core['remaining'] > 10  # Return True if we have enough requests
    return True

def generate_all_graphs():
    """Generate graphs for ALL repositories"""
    print("🚀 Generating ALL repository graphs...")
    # Check rate limit before starting
    if not check_rate_limit():
        print("⚠️ Rate limit too low, skipping graph generation")
        return 0
    try:
        repos = fetch_repositories()
        total_repos = len(repos)
        successful = 0
        failed = 0
        for i, repo in enumerate(repos, 1):
            repo_name = repo['name']
            graph_path = os.path.join(GRAPHS_FOLDER, f"{repo_name}_commits.png")
            stats_path = os.path.join(GRAPHS_FOLDER, f"{repo_name}_stats.json")
            print(f"📊 [{i}/{total_repos}] Generating graph for {repo_name}...")
            try:
                stats = create_commit_graph(repo_name, graph_path)
                if stats:
                    # Save stats alongside the graph
                    with open(stats_path, 'w') as f:
                        json.dump(stats, f, indent=2)
                    print(f"✅ [{i}/{total_repos}] Generated graph for {repo_name}")
                    successful += 1
                else:
                    print(f"⚠️ [{i}/{total_repos}] No commits found for {repo_name}")
                    failed += 1
                    
            except Exception as e:
                print(f"❌ [{i}/{total_repos}] Error generating {repo_name}: {e}")
                failed += 1
                time.sleep(0.1)  # 100ms between graphs
        print(f"📈 Graph generation complete: {successful} successful, {failed} failed")
        return successful
    except Exception as e:
        print(f"❌ Error in generate_all_graphs: {e}")
        return 0
# Added Flask configuration for better performance
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300  # Cache static files for 5 minutes
app.config['JSON_SORT_KEYS'] = False  # Preserve JSON key order

def process_commit_data(commit_dates):
    """Convert commit dates to daily counts with actual dates"""
    if not commit_dates:
        return [], []  # Return empty arrays
    
    # Parse dates
    dates = [datetime.fromisoformat(date.replace('Z', '+00:00')) for date in commit_dates]
    dates.sort()
    
    # Timeline from first commit to today
    first_date = dates[0].date()
    today = datetime.now(timezone.utc).date()
    
    print(f"Timeline: {first_date} to {today}")
    
    # Generate timeline from first commit to today
    timeline = pd.date_range(start=first_date, end=today, freq='D')
    
    # Count commits per day
    commit_counts = {}
    for date in dates:
        day = date.date()
        commit_counts[day] = commit_counts.get(day, 0) + 1
    
    # Create date-based x, y data
    x_dates = []  # Actual dates
    y = []        # Commits per day
    
    for date in timeline:
        day_date = date.date()
        commits = commit_counts.get(day_date, 0)
        x_dates.append(date.to_pydatetime())  # Convert to datetime for matplotlib
        y.append(commits)
    
    print(f"Timeline: {len(timeline)} days from first commit to today")
    print(f"Total commits: {sum(y)}, Active days: {sum(1 for c in y if c > 0)}")
    
    return x_dates, y

def create_commit_graph(repo_name, save_path):
    """Create seaborn regplot with dates on x-axis and LOWESS smoothing"""
    # Fetch and process commit data
    print(f"🚀 Starting graph creation for {repo_name}...")
    commit_dates = fetch_commits(repo_name)
    if not commit_dates:
        print(f"❌ No commits found for {repo_name}")
        return None
    
    x_dates, y = process_commit_data(commit_dates)
    if not x_dates or not y:
        print(f"❌ Failed to process commit data for {repo_name}")
        return None
    
    print(f"📈 Timeline data ready for {repo_name}: {len(x_dates)} data points")
    
    # Calculate statistics
    total_commits, active_days, timeline_length = sum(y), sum(1 for c in y if c > 0), len(x_dates)
    baseline = 0.1 if timeline_length > 400 else 0.05 if timeline_length > 200 else 0.01 + (timeline_length // 100) * 0.01
    
    # Create plot
    plt.figure(figsize=(12, 6), facecolor='white')
    ax = plt.gca()
    sns.set_style("darkgrid")
    
    # Scatter points and smoothed curve
    plt.scatter(x_dates, y, color='blue', alpha=0.9, s=60, edgecolors='white', linewidth=2, zorder=3)
    
    # Create and apply smoothing
    y_smooth = np.full_like(y, baseline, dtype=float)
    for i in range(len(y)):
        if y[i] > 0:
            y_smooth[i] = max(baseline, y[i])
    
    y_smooth = gaussian_filter1d(y_smooth, sigma=0.8)
    y_smooth = np.maximum(y_smooth, baseline)
    
    for i in range(len(y)):
        if y[i] > 0:
            y_smooth[i] = max(y_smooth[i], y[i] * 0.8, baseline * 3)
    
    y_final = np.maximum(gaussian_filter1d(y_smooth, sigma=0.4), baseline)
    plt.plot(x_dates, y_final, color='red', linewidth=4, alpha=1.0, zorder=2)
    
    # Set axis limits
    max_val = max(max(y) if y else 1, max(y_final) if len(y_final) > 0 else 1)
    plt.ylim(-0.05, max_val * 1.05)
    plt.xlim(x_dates[0], x_dates[-1])
    
    # Date formatter
    def date_formatter(x, pos):
        try:
            date = mdates.num2date(x)
            return date.strftime('%-m/%-d/%y')
        except:
            try:
                return date.strftime('%#m/%#d/%y')
            except:
                return date.strftime('%m/%d/%y').lstrip('0').replace('/0', '/')
    
    ax.xaxis.set_major_formatter(FuncFormatter(date_formatter))
    
    # Smart tick placement
    first_commit_date, last_date = x_dates[0], x_dates[-1]
    tick_positions = [first_commit_date]
    
    if timeline_length <= 30:
        # Short projects: every timeline_length/5 days
        interval_days = math.ceil(timeline_length / 5)
        current_date = first_commit_date
        while current_date <= last_date:
            tick_positions.append(current_date)
            current_date += timedelta(days=interval_days)
    elif timeline_length <= 90:
        # 1-3 months: first, middle, end of months
        current_date = first_commit_date
        next_month = current_date.replace(day=1) + relativedelta(months=1) if current_date.day > 1 else current_date
        
        while next_month <= last_date:
            tick_positions.append(next_month)  # First of month
            days_in_month = calendar.monthrange(next_month.year, next_month.month)[1]
            middle_date = next_month.replace(day=days_in_month // 2)
            end_date = next_month.replace(day=days_in_month)
            
            if middle_date <= last_date and (30 > timeline_length or abs((last_date - middle_date).days) > 10):
                tick_positions.append(middle_date)
            if end_date <= last_date and (30 > timeline_length or abs((last_date - end_date).days) > 10):
                tick_positions.append(end_date)
            
            next_month += relativedelta(months=1)
    else:
        # 3+ months: monthly (first of each month)
        next_month = first_commit_date.replace(day=1) + relativedelta(months=1) if first_commit_date.day > 1 else first_commit_date
        monthly_dates = []
        while next_month <= last_date:
            monthly_dates.append(next_month)
            next_month += relativedelta(months=1)
        
        # Filter out recent months if too close to end date
        if timeline_length <= 180:
            tick_positions.extend([d for d in monthly_dates if abs((last_date - d).days) > 10])
        else:
            tick_positions.extend(monthly_dates[:-1])  # Remove most recent month
    
    # Add final date if not included
    if tick_positions[-1] != last_date:
        tick_positions.append(last_date)
    
    # Set ticks and labels
    ax.set_xticks(sorted(list(set(tick_positions))))
    plt.xticks(rotation=0, ha='center')
    
    # Timeline description
    if timeline_length <= 7:
        timeline_desc = f"({timeline_length} days)"
    elif timeline_length <= 60:
        timeline_desc = f"({timeline_length} days, ~{timeline_length//7} weeks)"
    else:
        timeline_desc = f"({timeline_length} days, ~{timeline_length//30} months)"
    
    # Labels and formatting
    plt.title(f'Commit Timeline for {repo_name} {timeline_desc}', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=12, labelpad=10)
    plt.ylabel('Commits per day', fontsize=12, labelpad=10)
    
    # Repository stats
    avg_commits = total_commits / active_days if active_days > 0 else 0
    density = active_days / timeline_length * 100 if timeline_length > 0 else 0
    
    def format_date(date_obj):
        try:
            return date_obj.strftime('%-m/%-d/%y')
        except:
            try:
                return date_obj.strftime('%#m/%#d/%y')
            except:
                formatted = date_obj.strftime('%m/%d/%y')
                return formatted.lstrip('0').replace('/0', '/')
    
    start_date, end_date = format_date(x_dates[0]), format_date(x_dates[-1])
    plt.figtext(0.02, 0.02, 
               f'Total: {total_commits} commits | Active: {active_days}/{timeline_length} days ({density:.1f}%) | Period: {start_date} to {end_date}',
               fontsize=10, ha='left')
    
    # Save and return
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05,
                facecolor='white', edgecolor='none', transparent=False)
    plt.close()
    
    print(f"Graph saved to {save_path}")
    
    return {
        'total_commits': total_commits,
        'active_days': active_days,
        'avg_commits_per_day': round(avg_commits, 1),
        'timeline_days': len(x_dates),
        'activity_density': round(density, 1),
        'date_range': f"{start_date} to {end_date}"
    }

# NEW PRELOADING SYSTEM FUNCTIONS
def generate_all_graphs():
    """Generate graphs for ALL repositories"""
    print("🚀 Generating ALL repository graphs...")
    
    try:
        repos = fetch_repositories()
        total_repos = len(repos)
        successful = 0
        failed = 0
        
        print(f"📊 Found {total_repos} repositories to process")
        
        for i, repo in enumerate(repos, 1):
            repo_name = repo['name']
            graph_path = os.path.join(GRAPHS_FOLDER, f"{repo_name}_commits.png")
            stats_path = os.path.join(GRAPHS_FOLDER, f"{repo_name}_stats.json")
            
            print(f"📊 [{i}/{total_repos}] Generating graph for {repo_name}...")
            
            try:
                stats = create_commit_graph(repo_name, graph_path)
                
                if stats:
                    # Save stats alongside the graph
                    with open(stats_path, 'w') as f:
                        json.dump(stats, f, indent=2)
                    print(f"✅ [{i}/{total_repos}] Generated graph for {repo_name}")
                    successful += 1
                else:
                    print(f"⚠️ [{i}/{total_repos}] No commits found for {repo_name}")
                    failed += 1
                    
            except Exception as e:
                print(f"❌ [{i}/{total_repos}] Error generating {repo_name}: {e}")
                failed += 1
                
            # Small delay between requests to be nice to GitHub API
            if i < total_repos:
                time.sleep(0.1)  # 100ms between graphs
        
        print(f"📈 Graph generation complete: {successful} successful, {failed} failed out of {total_repos} repositories")
        return successful
        
    except Exception as e:
        print(f"❌ Error in generate_all_graphs: {e}")
        return 0

def hourly_graph_refresh():
    print("🕐 Starting background graph generation system...")
    print(f"🌍 Server timezone: {datetime.now()}")
    print(f"🌍 UTC time: {datetime.now(timezone.utc)}")
    
    # Generate all graphs immediately on startup
    generate_all_graphs()
    
    while True:
        try:
            print(f"⏰ Waiting 1 hour for next refresh... (next update at {(datetime.now() + timedelta(hours=1)).strftime('%H:%M:%S')})")
            time.sleep(3600)
            
            print("🔄 HOURLY REFRESH: Regenerating all graphs...")
            print(f"🕐 Current time: {datetime.now()}")
            successful = generate_all_graphs()

            if successful > 0:
                print(f"✅ Hourly refresh completed successfully ({successful} graphs updated)")
            else:
                print("⚠️ Hourly refresh completed but no graphs were generated")
            
        except Exception as e:
            print(f"❌ Error in hourly refresh cycle: {e}")
            # Continue the loop even if there's an error
            time.sleep(60)  # Wait 1 minute before retrying

def start_background_graph_system():
    """Start the background graph generation system"""
    refresh_thread = threading.Thread(target=hourly_graph_refresh, daemon=True)
    refresh_thread.start()
    print("🚀 Background graph system started - generating all graphs every hour")

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
    """Get list of repositories with preloaded graph info"""
    try:
        repos = fetch_repositories()
        total_repos = len(repos)
        
        repo_data = []
        for repo in repos:
            repo_name = repo['name']
            graph_filename = f"{repo_name}_commits.png"
            graph_path = os.path.join(GRAPHS_FOLDER, graph_filename)
            stats_filename = f"{repo_name}_stats.json"
            stats_path = os.path.join(GRAPHS_FOLDER, stats_filename)
            
            # Check if BOTH graph and stats files exist
            graph_info = None
            if os.path.exists(graph_path) and os.path.exists(stats_path):
                try:
                    # Check file age (for display purposes)
                    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(graph_path))
                    age_minutes = int(file_age.total_seconds() / 60)
                    
                    # Load stats
                    with open(stats_path, 'r') as f:
                        stats = json.load(f)
                    
                    graph_info = {
                        'image_url': f"/static/graphs/{graph_filename}",
                        'stats': stats,
                        'ready': True,
                        'age_minutes': age_minutes
                    }
                    print(f"✅ Preloaded graph available for {repo_name} (age: {age_minutes}m)")
                    
                except Exception as e:
                    print(f"⚠️ Error loading cached data for {repo_name}: {e}")
                    graph_info = {'ready': False, 'reason': 'corrupted'}
            else:
                print(f"❌ No preloaded graph for {repo_name}")
                graph_info = {'ready': False, 'reason': 'missing'}
            
            repo_data.append({
                'name': repo['name'],
                'description': repo['description'],
                'stars': repo['stargazers_count'],
                'forks': repo['forks_count'],
                'language': repo['language'],
                'updated_at': repo['updated_at'],
                'html_url': repo['html_url'],
                'graph': graph_info
            })
        
        ready_count = sum(1 for repo in repo_data if repo['graph']['ready'])
        print(f"📊 Repository API called: {ready_count}/{total_repos} graphs ready")
        
        return jsonify({
            "status": "success",
            "repositories": repo_data,
            "graphs_ready": ready_count,
            "total_repos": len(repo_data)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/generate_graph/<repo_name>')
def generate_graph(repo_name):
    """Generate commit graph for a specific repository (with caching)"""
    try:
        graph_filename = f"{repo_name}_commits.png"
        graph_path = os.path.join(GRAPHS_FOLDER, graph_filename)
        stats_filename = f"{repo_name}_stats.json"
        stats_path = os.path.join(GRAPHS_FOLDER, stats_filename)
        
        # Check if both graph and stats files exist
        if os.path.exists(graph_path) and os.path.exists(stats_path):
            print(f"Using cached graph and stats for {repo_name}")
            
            # Load cached stats
            try:
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                
                return jsonify({
                    "status": "success",
                    "image_url": f"/static/graphs/{graph_filename}",
                    "stats": stats,
                    "cached": True
                })
            except json.JSONDecodeError:
                print(f"Corrupted stats file for {repo_name}, regenerating...")
                # Falls through to regeneration
        
        # Generate new graph if it doesn't exist or stats are corrupted
        print(f"Generating new graph for {repo_name}")
        stats = create_commit_graph(repo_name, graph_path)
        
        if stats is not None:
            # Save stats alongside the image
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            return jsonify({
                "status": "success",
                "image_url": f"/static/graphs/{graph_filename}",
                "stats": stats,
                "cached": False
            })
        else:
            return jsonify({
                "status": "error",
                "message": "No commits found or error generating graph"
            }), 404
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/generate_top_graphs/<int:top_n>')
def generate_top_graphs(top_n=3):
    """Generate graphs for top N repositories"""
    try:
        # Get repositories and their commit counts
        repos = fetch_repositories()
        repo_data = []
        
        for repo in repos[:top_n * 2]:  # Fetch extra in case some have no commits
            commits = fetch_commits(repo['name'])
            if commits:
                x_dates, y = process_commit_data(commits)
                total_commits = sum(y) if y else 0
                repo_data.append({
                    'name': repo['name'],
                    'commits': total_commits,
                    'repo_info': repo
                })
        
        # Sort by commit count and take top N
        repo_data.sort(key=lambda x: x['commits'], reverse=True)
        top_repos = repo_data[:top_n]
        
        # Generate individual graphs for each top repo
        results = []
        for repo_info in top_repos:
            repo_name = repo_info['name']
            graph_filename = f"{repo_name}_commits.png"
            graph_path = os.path.join(GRAPHS_FOLDER, graph_filename)
            
            # Create individual graph
            stats = create_commit_graph(repo_name, graph_path)
            
            if stats:
                results.append({
                    "repo_name": repo_name,
                    "image_url": f"/static/graphs/{graph_filename}",
                    "total_commits": stats['total_commits'],
                    "description": repo_info['repo_info']['description'],
                    "date_range": stats.get('date_range', 'N/A')
                })
        
        return jsonify({
            "status": "success",
            "graphs": results
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/static/graphs/<filename>')
def serve_graph(filename):
    """Serve generated graph images"""
    return send_from_directory(GRAPHS_FOLDER, filename)

@app.route('/api/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "graphs_folder": GRAPHS_FOLDER
    })

if __name__ == '__main__':
    print("🚀 Starting GitHub Graphs Server...")
    # PRE-BUILD: Generate all graphs FIRST
    print("📊 Pre-building all graphs before starting server...")
    successful = generate_all_graphs()
    print(f"✅ Pre-build complete: {successful} graphs generated")
    # Start background system BEFORE app.run()
    start_background_graph_system()
    # Start Flask app (this blocks, so put it last)
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    print(f"🌐 Server starting on port {port}")
    app.run(debug=debug, host='0.0.0.0', port=port)