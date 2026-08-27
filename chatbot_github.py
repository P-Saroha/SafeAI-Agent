"""
GitHub Repository Analysis Tool
Simple code to analyze GitHub repos and understand what they do
"""

import re
import os
import logging
from typing import Optional, Dict, List
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Extract GitHub URL from text
# ══════════════════════════════════════════════════════════════════════════

def extract_github_url(text: str) -> Optional[str]:
    """
    Find a GitHub URL in the user's message.
    
    Examples:
    - "Check https://github.com/langchain-ai/langchain" → returns URL
    - "github.com/facebook/react" → returns full URL
    """
    # Try to find https://github.com/owner/repo format
    pattern1 = r'https?://github\.com/([^/\s]+)/([^/\s]+)/?'
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        return f"https://github.com/{match.group(1)}/{match.group(2)}"
    
    # Try to find github.com/owner/repo format (without http)
    pattern2 = r'github\.com/([^/\s]+)/([^/\s]+)/?'
    match = re.search(pattern2, text, re.IGNORECASE)
    if match:
        return f"https://github.com/{match.group(1)}/{match.group(2)}"
    
    return None


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Parse owner and repo name from URL
# ══════════════════════════════════════════════════════════════════════════

def parse_github_url(url: str) -> Optional[tuple]:
    """
    Break down URL into owner and repo name.
    
    Example:
    - "https://github.com/langchain-ai/langchain"
    - Returns: ("langchain-ai", "langchain")
    """
    match = re.search(r'github\.com/([^/]+)/([^/]+)', url, re.IGNORECASE)
    
    if match:
        owner = match.group(1)
        repo = match.group(2)
        return (owner, repo)
    
    return None


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Main function - analyze a repo
# ══════════════════════════════════════════════════════════════════════════

def analyze_github_repo(github_url: str) -> Dict:
    """
    Main function: Connect to GitHub API and get all repo information.
    
    Returns a dictionary with all repo details, or error message.
    """
    
    # Import GitHub library
    from github import Github
    
    # Parse the URL
    owner_repo = parse_github_url(github_url)
    if not owner_repo:
        return {"error": "Invalid GitHub URL format"}
    
    owner, repo_name = owner_repo
    print(f"[GITHUB] Analyzing: {owner}/{repo_name}")
    
    try:
        # Create GitHub connection
        # If user has GITHUB_TOKEN in .env, use it (5000 requests/hour)
        # Otherwise use public API (60 requests/hour)
        token = os.getenv("GITHUB_TOKEN")
        
        if token:
            print(f"[GITHUB] Using authenticated API (5000 req/hour)")
            github_client = Github(token)
        else:
            print(f"[GITHUB] Using public API (60 req/hour)")
            github_client = Github()
        
        # Get the repository object
        repo = github_client.get_user(owner).get_repo(repo_name)
        
        # Collect basic information
        result = {
            "url": repo.html_url,
            "name": repo.name,
            "owner": repo.owner.login,
            "description": repo.description or "No description",
            "language": repo.language or "Unknown",
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "created_at": str(repo.created_at)[:10],
            "last_updated": str(repo.updated_at)[:10],
            "license": repo.license.name if repo.license else "None",
            "homepage": repo.homepage,
            "topics": repo.get_topics(),
            "open_issues": repo.open_issues_count,
        }
        
        # Get optional data
        result["readme"] = get_readme(repo)
        result["languages"] = get_languages(repo)
        result["structure"] = get_structure(repo)
        result["purpose"] = infer_purpose(repo)
        
        print(f"[GITHUB] Success!")
        return result
        
    except Exception as e:
        # If something goes wrong, return error message
        error_msg = str(e)
        return {"error": f"Failed to analyze: {error_msg}"}


# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS - Get different types of data
# ══════════════════════════════════════════════════════════════════════════

def get_readme(repo) -> Optional[str]:
    """
    Try to fetch the README file from the repository.
    
    Returns the full content, or None if not found.
    """
    try:
        readme_file = repo.get_readme()
        content = readme_file.decoded_content.decode('utf-8')
        return content
    except:
        # README doesn't exist, return None
        return None


def get_languages(repo) -> Dict[str, float]:
    """
    Get the programming languages used in the repo.
    
    Returns a dictionary like: {"Python": 99.3, "Makefile": 0.5}
    Shows what percentage each language is.
    """
    try:
        # Get languages from repo
        languages_dict = repo.get_languages()
        
        if not languages_dict:
            return {}
        
        # Calculate total bytes
        total_bytes = sum(languages_dict.values())
        
        # Create result dictionary
        result = {}
        
        # Sort by size (biggest first) and take top 5
        sorted_languages = sorted(
            languages_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Calculate percentage for each language
        for language_name, byte_count in sorted_languages:
            percentage = (byte_count / total_bytes) * 100
            percentage = round(percentage, 1)
            result[language_name] = percentage
        
        return result
    except:
        return {}


def get_structure(repo) -> Dict[str, List[str]]:
    """
    Get top-level folders and files in the repository.
    
    Returns a dictionary with "directories" and "files" keys.
    """
    try:
        # Get contents of repo root
        repo_contents = repo.get_contents("")
        
        # Separate into folders and files
        directories = []
        files = []
        
        for item in repo_contents:
            if item.type == "dir":
                directories.append(item.name)
            elif item.type == "file":
                files.append(item.name)
        
        # Keep only first 10 of each
        directories = directories[:10]
        files = files[:10]
        
        return {
            "directories": directories,
            "files": files
        }
    except:
        return {"directories": [], "files": []}


def infer_purpose(repo) -> Dict[str, str]:
    """
    Figure out what the project does WITHOUT reading README.
    
    Uses: topics, description, language, stars to guess the project type.
    """
    
    inference = {
        "type": "Unknown",
        "tags": [],
        "maturity": "Unknown"
    }
    
    try:
        # Get repo topics (tags like "python", "ai", "api", etc)
        topics = repo.get_topics()
        
        if topics:
            # Save first 5 topics
            inference["tags"] = topics[:5]
            
            # Combine all topics into one string
            topics_text = " ".join(topics).lower()
            
            # Check what type of project this is based on topics
            if any(word in topics_text for word in ["api", "rest", "backend"]):
                inference["type"] = "Backend/API"
            elif any(word in topics_text for word in ["frontend", "react", "vue"]):
                inference["type"] = "Frontend"
            elif any(word in topics_text for word in ["ai", "ml", "llm", "rag"]):
                inference["type"] = "AI/ML"
            elif any(word in topics_text for word in ["library", "framework", "sdk"]):
                inference["type"] = "Library/Framework"
            elif any(word in topics_text for word in ["cli", "tool", "devops"]):
                inference["type"] = "DevOps/CLI"
            elif any(word in topics_text for word in ["database", "data"]):
                inference["type"] = "Data/Database"
        
        # Check maturity based on stars
        stars = repo.stargazers_count
        if stars > 10000:
            inference["maturity"] = "Very Popular (10K+ stars)"
        elif stars > 1000:
            inference["maturity"] = "Popular (1K+ stars)"
        elif stars > 100:
            inference["maturity"] = "Growing (100+ stars)"
        else:
            inference["maturity"] = "Early Stage"
        
        return inference
    except:
        return inference


# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Format the response nicely
# ══════════════════════════════════════════════════════════════════════════

def format_github_response(info: Dict) -> str:
    """
    Take all the data we collected and format it into a nice response.
    """
    
    # Check if there was an error
    if "error" in info:
        return f"Error: {info['error']}"
    
    # Get AI summary if README exists
    groq_summary = ""
    if info.get("readme"):
        groq_summary = get_groq_summary(info)
    
    # Build the response text
    response = "GitHub Repository Analysis\n"
    response += "=" * 60 + "\n\n"
    
    # Basic info
    response += f"Name: {info['name']}\n"
    response += f"Owner: {info['owner']}\n"
    response += f"URL: {info['url']}\n\n"
    
    # Description
    response += f"Description:\n{info['description']}\n\n"
    
    # Quick analysis (from metadata, no README needed)
    purpose = info.get("purpose", {})
    if purpose.get("type") != "Unknown":
        response += "Quick Analysis (from metadata):\n"
        response += f"  Type: {purpose.get('type')}\n"
        if purpose.get("tags"):
            response += f"  Tags: {', '.join(purpose['tags'])}\n"
        response += f"  Maturity: {purpose.get('maturity')}\n"
        response += "\n"
    
    # Stats
    response += f"Stats:\n"
    response += f"  Stars: {info['stars']:,}\n"
    response += f"  Forks: {info['forks']:,}\n"
    response += f"  Created: {info['created_at']}\n"
    response += f"  Updated: {info['last_updated']}\n"
    response += f"  Open Issues: {info['open_issues']}\n\n"
    
    # Languages
    if info.get("languages"):
        response += "Languages Used:\n"
        for language, percentage in info["languages"].items():
            response += f"  {language}: {percentage}%\n"
        response += "\n"
    
    # Topics
    if info.get("topics"):
        topics_list = ", ".join(info["topics"][:8])
        response += f"Topics: {topics_list}\n"
        response += f"License: {info['license']}\n\n"
    
    # Directory structure
    response += "Project Structure (Top Level):\n"
    
    dirs = info["structure"].get("directories", [])
    if dirs:
        response += f"  Folders: {', '.join(dirs[:5])}\n"
    else:
        response += f"  Folders: None\n"
    
    files = info["structure"].get("files", [])
    if files:
        response += f"  Files: {', '.join(files[:5])}\n"
    else:
        response += f"  Files: None\n"
    
    response += "\n"
    
    # Add Groq AI summary if available
    if groq_summary:
        response += "=" * 60 + "\n"
        response += "AI ANALYSIS (Groq)\n"
        response += "=" * 60 + "\n"
        response += groq_summary + "\n\n"
    
    # Add full README if available
    if info.get("readme"):
        response += "=" * 60 + "\n"
        response += "README\n"
        response += "=" * 60 + "\n"
        response += info["readme"] + "\n"
    
    return response


# ══════════════════════════════════════════════════════════════════════════
# OPTIONAL: Use Groq LLM to generate smart summary
# ══════════════════════════════════════════════════════════════════════════

def get_groq_summary(info: Dict) -> str:
    """
    Use Groq LLM to analyze the README and create a structured summary.
    
    This is optional - works only if GROQ_API_KEY is set.
    """
    try:
        from langchain_openai import ChatOpenAI
        
        # Create LLM connection
        llm = ChatOpenAI(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            temperature=0,
            timeout=10,
        )
        
        # Get first 1500 characters of README
        readme_preview = (info.get("readme") or "")[:1500]
        
        # Get first 5 topics
        topics = ", ".join(info.get("topics", [])[:5])
        
        # Create prompt for Groq
        prompt = f"""Analyze this repository and provide a structured summary.

Repository Name: {info['name']}
Stars: {info['stars']}
Language: {info['language']}
Description: {info['description']}
Topics: {topics}

README Content:
{readme_preview}

Please provide your response in this format:

WHAT IS IT?
[Brief explanation - 1-2 sentences]

USE CASE
[Who should use this and why]

KEY FEATURES
- Feature 1
- Feature 2
- Feature 3

TECH STACK
[Main technologies used]

GETTING STARTED
[Quick start instructions]

Summary:"""
        
        # Call Groq
        response = llm.invoke(prompt)
        
        # Extract text from response
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
        
    except Exception as e:
        # If Groq fails, just return empty string (graceful fallback)
        logger.debug(f"Groq summary failed: {e}")
        return ""
