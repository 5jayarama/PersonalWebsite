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
import os

# MATPLOTLIB FIX: Set backend before importing matplotlib.pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

app = Flask(__name__)
CORS(app)  # Enable CORS for web requests

# Configure paths
STATIC_FOLDER = 'static'
GRAPHS_FOLDER = os.path.join(STATIC_FOLDER, 'graphs')
GITHUB_USERNAME = '5jayarama'

# Ensure directories exist
os.makedirs(GRAPHS_FOLDER, exist_ok=True)

def fetch_repositories():
    """Fetch all repositories for the user"""
    url = f"https://api.github.com/users/{GITHUB_USERNAME}/repos"
    params = {'sort': 'updated', 'per_page': 100}
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching repositories: {response.status_code}")
        return []

def fetch_commits(repo_name, per_page=100):
    """Fetch commits for a specific repository"""
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/commits"
    params = {'per_page': per_page}
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        commits = response.json()
        return [commit['commit']['author']['date'] for commit in commits]
    else:
        print(f"Error fetching commits for {repo_name}: {response.status_code}")
        return []

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
    
    # Format date range for display in M/D/YY format (no leading zeros)
    def format_date_no_zeros(date_obj):
        """Format date without leading zeros"""
        try:
            return date_obj.strftime('%-m/%-d/%y')  # Unix format
        except:
            try:
                return date_obj.strftime('%#m/%#d/%y')  # Windows format
            except:
                # Fallback: manual removal
                formatted = date_obj.strftime('%m/%d/%y')
                return formatted.lstrip('0').replace('/0', '/')
    
    start_date = format_date_no_zeros(x_dates[0])
    end_date = format_date_no_zeros(x_dates[-1])
    
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

@app.route('/')
def index():
    """Serve the main HTML page"""
    # Try multiple possible HTML file names
    possible_files = ['website2.html', 'index.html', 'website.html']
    
    for filename in possible_files:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            continue
    
    # If no HTML file found, show setup instructions
    return '''
    <h1>Setup Required</h1>
    <p>Please save your HTML file as one of these names in the same folder as server.py:</p>
    <ul>
        <li>website2.html</li>
        <li>index.html</li>
        <li>website.html</li>
    </ul>
    <p>Current folder should have:</p>
    <pre>
    PersonalWebsite/
    ├── server.py
    └── [your-html-file].html
    </pre>
    '''

@app.route('/api/repositories')
def get_repositories():
    """Get list of repositories"""
    try:
        repos = fetch_repositories()
        
        # Show all repositories
        repo_data = []
        for repo in repos[:15]:  # Reasonable limit of 15 instead of 5
            repo_data.append({
                'name': repo['name'],
                'description': repo['description'],
                'stars': repo['stargazers_count'],
                'forks': repo['forks_count'],
                'language': repo['language'],
                'updated_at': repo['updated_at'],
                'html_url': repo['html_url'],
                'commit_count': 'Click to generate'  # Don't fetch upfront for speed
            })
        
        return jsonify({
            "status": "success",
            "repositories": repo_data
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/generate_graph/<repo_name>')
def generate_graph(repo_name):
    """Generate commit graph for a specific repository"""
    try:
        # Generate graph and save to static folder
        graph_filename = f"{repo_name}_commits.png"
        graph_path = os.path.join(GRAPHS_FOLDER, graph_filename)
        
        # Create the graph
        stats = create_commit_graph(repo_name, graph_path)
        
        if stats is not None:
            return jsonify({
                "status": "success",
                "image_url": f"/static/graphs/{graph_filename}",
                "stats": stats
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

if __name__ == '__main__' and os.environ.get("RENDER") != "true":
    app.run(debug=True, host='0.0.0.0', port=5000)