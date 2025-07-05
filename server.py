from flask import Flask, render_template, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
from dateutil.relativedelta import relativedelta
import calendar
import math
import threading
import time
from flask import send_file

# MATPLOTLIB FIX: Set backend before importing matplotlib.pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

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
    
    response = requests.get(url, params=params, headers=GITHUB_HEADERS)
    if response.status_code == 200:
        commits = response.json()
        return [commit['commit']['author']['date'] for commit in commits]
    else:
        print(f"Error fetching commits for {repo_name}: {response.status_code}")
        if response.status_code == 403:
            print(f"Rate limit info: {response.headers.get('X-RateLimit-Remaining', 'unknown')} requests remaining")
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
        successful = 0
        failed = 0
        for i, repo in enumerate(repos[:15], 1):
            repo_name = repo['name']
            graph_path = os.path.join(GRAPHS_FOLDER, f"{repo_name}_commits.png")
            stats_path = os.path.join(GRAPHS_FOLDER, f"{repo_name}_stats.json")
            print(f"📊 [{i}/15] Generating graph for {repo_name}...")
            try:
                stats = create_commit_graph(repo_name, graph_path)
                if stats:
                    # Save stats alongside the graph
                    with open(stats_path, 'w') as f:
                        json.dump(stats, f, indent=2)
                    print(f"✅ [{i}/15] Generated graph for {repo_name}")
                    successful += 1
                else:
                    print(f"⚠️ [{i}/15] No commits found for {repo_name}")
                    failed += 1
                    
            except Exception as e:
                print(f"❌ [{i}/15] Error generating {repo_name}: {e}")
                failed += 1
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
    today = datetime.now().date()
    
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
    
    # Fetch commit data
    print(f"Fetching commits for {repo_name}...")
    commit_dates = fetch_commits(repo_name)
    
    if not commit_dates:
        print(f"No commits found for {repo_name}")
        return None
    
    # Process data into date-based x, y format
    x_dates, y = process_commit_data(commit_dates)
    
    if not x_dates or not y:
        return None
    
    # Calculate statistics
    total_commits = sum(y)
    active_days = sum(1 for commits in y if commits > 0)
    timeline_length = len(x_dates)
    
    # Dynamic baseline for smoothing curve visibility
    if timeline_length > 400:
        baseline = 0.1
    elif timeline_length > 200:
        baseline = 0.05
    else:
        baseline = 0.01 + (timeline_length // 100) * 0.01
    print(f"Timeline {timeline_length} days -> baseline {baseline} for {repo_name}")
    
    # Create clean plot - back to original size
    plt.figure(figsize=(12, 6), facecolor='white')
    ax = plt.gca()
    sns.set_style("darkgrid")
    
    # Plot scatter points
    plt.scatter(x_dates, y, color='blue', alpha=0.9, s=60, edgecolors='white', linewidth=2, zorder=3)
    
    # Create smoothed curve
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    
    # Create baseline everywhere, then add peaks
    y_smooth = np.full_like(y, baseline, dtype=float)
    
    # Add actual commit data on top of baseline
    for i in range(len(y)):
        if y[i] > 0:
            y_smooth[i] = max(baseline, y[i])
    
    # Apply light smoothing
    y_smooth = gaussian_filter1d(y_smooth, sigma=0.8)
    
    # Force baseline everywhere
    y_smooth = np.maximum(y_smooth, baseline)
    
    # Enhance peaks but keep baseline intact
    for i in range(len(y)):
        if y[i] > 0:
            y_smooth[i] = max(y_smooth[i], y[i] * 0.8, baseline * 3)
    
    # Final gentle smoothing
    y_final = gaussian_filter1d(y_smooth, sigma=0.4)
    y_final = np.maximum(y_final, baseline)
    
    # Plot the smoothed curve
    plt.plot(x_dates, y_final, color='red', linewidth=4, alpha=1.0, zorder=2)
    
    # Set clean axis limits - start exactly at first commit date
    max_val = max(max(y) if y else 1, max(y_final) if len(y_final) > 0 else 1)
    plt.ylim(bottom=-0.05, top=max_val * 1.05)  # Move 0 slightly up from bottom
    plt.xlim(left=x_dates[0], right=x_dates[-1])  # Exact date range
    
    # Format x-axis with dates in M/D/YY format (no leading zeros) - HORIZONTAL
    from matplotlib.ticker import FuncFormatter
    
    def date_formatter(x, pos):
        """Custom formatter to remove leading zeros from dates"""
        try:
            date = mdates.num2date(x)
            return date.strftime('%-m/%-d/%y')  # %-m and %-d remove leading zeros on Unix
        except:
            try:
                return date.strftime('%#m/%#d/%y')  # %#m and %#d remove leading zeros on Windows
            except:
                return date.strftime('%m/%d/%y').lstrip('0').replace('/0', '/')
    
    ax.xaxis.set_major_formatter(FuncFormatter(date_formatter))
    
    # Intelligent tick placement based on timeline length
    first_commit_date = x_dates[0]
    last_date = x_dates[-1]
    
    # Generate tick positions based on timeline length
    tick_positions = []
    
    if timeline_length <= 30:
        # Short projects: label every timeline_length/5 days (rounded up)
        import math
        interval_days = math.ceil(timeline_length / 5)
        
        current_date = first_commit_date
        while current_date <= last_date:
            tick_positions.append(current_date)
            current_date += timedelta(days=interval_days)
    elif timeline_length <= 90:  # Up to 3 months
        # First of month, middle of month, end of month
        current_date = first_commit_date
        
        # Add the first commit date
        tick_positions.append(current_date)
        
        # Find the first of the next month
        if current_date.day > 1:
            next_month = current_date.replace(day=1) + relativedelta(months=1)
        else:
            next_month = current_date
        
        # Add monthly ticks
        monthly_ticks = []
        while next_month <= last_date:
            # First of month
            monthly_ticks.append(('first', next_month))
            
            # Middle of month (halfway point, rounded down)
            days_in_month = calendar.monthrange(next_month.year, next_month.month)[1]
            middle_day = days_in_month // 2
            middle_date = next_month.replace(day=middle_day)
            if middle_date <= last_date:
                monthly_ticks.append(('middle', middle_date))
            
            # End of month
            end_date = next_month.replace(day=days_in_month)
            if end_date <= last_date:
                monthly_ticks.append(('end', end_date))
            
            next_month += relativedelta(months=1)
        
        # Smart detection for 1-6 months: remove overlapping dates within 10 days
        if 30 <= timeline_length <= 180:  # 1-6 months
            filtered_ticks = []
            for tick_type, tick_date in monthly_ticks:
                # Check if this tick is too close to the final date (within 10 days)
                if abs((last_date - tick_date).days) > 10:
                    filtered_ticks.append(tick_date)
            tick_positions.extend(filtered_ticks)
        else:
            # Under 1 month: keep all ticks
            tick_positions.extend([tick_date for _, tick_date in monthly_ticks])
        
        # Add final date if not already included
        if tick_positions[-1] != last_date:
            tick_positions.append(last_date)
    else:
        # More than 3 months: just monthly (first of each month)
        current_date = first_commit_date
        
        # Add the first commit date
        tick_positions.append(current_date)
        
        # Find the first of the next month
        if current_date.day > 1:
            next_month = current_date.replace(day=1) + relativedelta(months=1)
        else:
            next_month = current_date
        
        # Add first of each month
        monthly_dates = []
        while next_month <= last_date:
            monthly_dates.append(next_month)
            next_month += relativedelta(months=1)
        
        if timeline_length <= 180:  # 3-6 months: smart detection (≤10 days)
            filtered_monthly = []
            for month_date in monthly_dates:
                if abs((last_date - month_date).days) > 10:
                    filtered_monthly.append(month_date)
            tick_positions.extend(filtered_monthly)
        else:  # Over 6 months: always remove most recent month
            if monthly_dates:
                tick_positions.extend(monthly_dates[:-1])  # Remove the most recent month
        
        # Add final date if not already included
        if tick_positions[-1] != last_date:
            tick_positions.append(last_date)
    
    # Remove duplicates and sort
    tick_positions = sorted(list(set(tick_positions)))
    
    # Set the tick locations
    ax.set_xticks(tick_positions)
    
    # Keep date labels horizontal
    plt.xticks(rotation=0, ha='center')
    
    # Add labels and title
    if timeline_length <= 7:
        timeline_desc = f"({timeline_length} days)"
    elif timeline_length <= 60:
        timeline_desc = f"({timeline_length} days, ~{timeline_length//7} weeks)"
    else:
        timeline_desc = f"({timeline_length} days, ~{timeline_length//30} months)"
    
    plt.title(f'Commit Timeline for {repo_name} {timeline_desc}', 
             fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=12, labelpad=10)
    plt.ylabel('Commits per day', fontsize=12, labelpad=10)
    
    # Add repository info
    avg_commits = total_commits / active_days if active_days > 0 else 0
    density = active_days / timeline_length * 100 if timeline_length > 0 else 0
    
    # Format date range for display in M/D/YY format with no leading zeros
    def format_date(date_obj):
        try:
            return date_obj.strftime('%-m/%-d/%y')  # Unix format
        except:
            try:
                return date_obj.strftime('%#m/%#d/%y')  # Windows format
            except:
                # Fallback: manual removal
                formatted = date_obj.strftime('%m/%d/%y')
                return formatted.lstrip('0').replace('/0', '/')
    
    start_date = format_date(x_dates[0])
    end_date = format_date(x_dates[-1])
    
    plt.figtext(0.02, 0.02, 
               f'Total: {total_commits} commits | Active: {active_days}/{timeline_length} days ({density:.1f}%) | Period: {start_date} to {end_date}',
               fontsize=10, ha='left')
    
    # Clean layout - back to original spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save with minimal padding - back to original
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
        successful = 0
        failed = 0
        
        for i, repo in enumerate(repos[:15], 1):
            repo_name = repo['name']
            graph_path = os.path.join(GRAPHS_FOLDER, f"{repo_name}_commits.png")
            stats_path = os.path.join(GRAPHS_FOLDER, f"{repo_name}_stats.json")
            
            print(f"📊 [{i}/15] Generating graph for {repo_name}...")
            
            try:
                stats = create_commit_graph(repo_name, graph_path)
                
                if stats:
                    # Save stats alongside the graph
                    with open(stats_path, 'w') as f:
                        json.dump(stats, f, indent=2)
                    print(f"✅ [{i}/15] Generated graph for {repo_name}")
                    successful += 1
                else:
                    print(f"⚠️ [{i}/15] No commits found for {repo_name}")
                    failed += 1
                    
            except Exception as e:
                print(f"❌ [{i}/15] Error generating {repo_name}: {e}")
                failed += 1
        
        print(f"📈 Graph generation complete: {successful} successful, {failed} failed")
        return successful
        
    except Exception as e:
        print(f"❌ Error in generate_all_graphs: {e}")
        return 0

def hourly_graph_refresh():
    """Background task that generates ALL graphs every hour"""
    print("🕐 Starting background graph generation system...")
    
    # Generate all graphs immediately on startup
    generate_all_graphs()
    
    while True:
        try:
            # Wait exactly 1 hour (3600 seconds)
            print(f"⏰ Waiting 1 hour for next refresh... (next update at {(datetime.now() + timedelta(hours=1)).strftime('%H:%M:%S')})")
            time.sleep(3600)
            
            print("🔄 HOURLY REFRESH: Regenerating all graphs...")
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
        
        repo_data = []
        for repo in repos[:15]:
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
        print(f"📊 Repository API called: {ready_count}/15 graphs ready")
        
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
    
    # Start the background graph generation system
    start_background_graph_system()
    
    # Start Flask app
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    
    print(f"🌐 Server starting on port {port}")
    print("📊 All 15 graphs will be generated immediately and refreshed every hour")
    print("🔥 Graphs will display instantly when users click on repositories")
    
    app.run(debug=debug, host='0.0.0.0', port=port)