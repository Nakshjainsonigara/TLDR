import os
from typing import TypedDict, Annotated, List
from langgraph.graph import Graph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from datetime import datetime
import re
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup
import asyncio
from dotenv import load_dotenv

# API Setup
load_dotenv()

# Initialize models
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Data Structures
class GraphState(TypedDict):
    news_query: Annotated[str, "Input query to extract news search parameters from."]
    num_searches_remaining: Annotated[int, "Number of articles to search for."]
    search_params: Annotated[dict, "Structured argument for the Tavily API."]
    past_searches: Annotated[List[dict], "List of search params already used."]
    articles_metadata: Annotated[list[dict], "Article metadata response from the Tavily API"]
    scraped_urls: Annotated[List[str], "List of urls already scraped."]
    num_articles_tldr: Annotated[int, "Number of articles to create TL;DR for."]
    potential_articles: Annotated[List[dict], "Article with full text to consider summarizing."]
    tldr_articles: Annotated[List[dict], "Selected article TL;DRs."]
    formatted_results: Annotated[str, "Formatted results to display."]

class TavilyParams(BaseModel):
    """Parameters for Tavily search."""
    query: str = Field(description="Search query to find relevant news articles")
    search_depth: str = Field(
        description="Search depth level", 
        default="basic"
    )
    max_results: int = Field(
        description="Number of results to return",
        default=5,
        ge=1,
        le=10
    )

    class Config:
        json_schema_extra = {
            "examples": [{
                "query": "latest AI developments",
                "search_depth": "advanced",
                "max_results": 5
            }]
        }

def generate_tavily_params(state: GraphState) -> GraphState:
    """Based on the query, generate Tavily search params."""
    try:
        prompt = (
            f"Create search parameters for finding news about: {state['news_query']}\n\n"
            "Return ONLY a JSON object with these exact fields:\n"
            "{\n"
            '    "query": "the search query",\n'
            '    "search_depth": "basic or advanced",\n'
            '    "max_results": 5\n'
            "}"
        )
        
        # Get response from LLM
        response = llm.invoke(prompt).content
        
        # Clean up the response to ensure valid JSON
        response = response.strip()
        if response.startswith("```json"):
            response = response.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON manually
        result = json.loads(response)
        
        # Update state with parsed parameters
        state["search_params"] = {
            "query": result["query"],
            "search_depth": result["search_depth"],
            "max_results": result["max_results"]
        }
        
        print(f"Generated search params: {state['search_params']}")
        
    except Exception as e:
        print(f"Error in parameter generation: {e}")
        # Fallback parameters
        state["search_params"] = {
            "query": state["news_query"],
            "search_depth": "basic",
            "max_results": 5
        }
    
    return state

def retrieve_articles_metadata(state: GraphState) -> GraphState:
    """Retrieve articles using Tavily Search."""
    try:
        print(f"Searching with params: {state['search_params']}")
        search_results = tavily_client.search(
            query=state["search_params"]["query"],
            search_depth=state["search_params"]["search_depth"],
            max_results=state["search_params"]["max_results"]
        )
        
        articles = []
        for result in search_results['results']:
            # Add error handling for missing fields
            article = {
                'url': result.get('url', ''),
                'title': result.get('title', 'No title available'),
                'description': result.get('description', 'No description available')
            }
            if article['url']:  # Only add if URL exists
                articles.append(article)
        
        print(f"Found {len(articles)} articles")    
        state["articles_metadata"] = articles
        state["past_searches"].append(state["search_params"])
        
    except Exception as e:
        print(f"Error in retrieve_articles_metadata: {e}")
        state["articles_metadata"] = []
    
    return state

def format_results(state: GraphState) -> GraphState:
    """Format final results."""
    if not state["tldr_articles"]:
        state["formatted_results"] = "No relevant articles found."
        return state
        
    searches = [params["query"] for params in state["past_searches"]]
    formatted_results = f"Results for: {state['news_query']}\nSearches: {', '.join(searches)}\n\n"
    
    for article in state["tldr_articles"]:
        if "summary" in article:
            formatted_results += f"{article['summary']}\n\n"
        else:
            formatted_results += f"{article['title']}\n{article['url']}\n* No summary available\n\n"
    
    state["formatted_results"] = formatted_results
    print("Results formatted successfully")
    return state

def articles_text_decision(state: GraphState) -> str:
    """Determine next step based on article results."""
    if state["num_searches_remaining"] == 0:
        if not state["potential_articles"]:
            print("No articles found and no searches remaining.")
            return "END"
        else:
            print("No searches remaining, proceeding to select top URLs.")
            return "select_top_urls"
    
    if len(state["potential_articles"]) < state["num_articles_tldr"]:
        state["num_searches_remaining"] -= 1
        print(f"Not enough articles, searching again. Remaining searches: {state['num_searches_remaining']}")
        return "generate_tavily_params"
    
    print(f"Found enough articles ({len(state['potential_articles'])}), proceeding to select top URLs.")
    return "select_top_urls"

def retrieve_articles_text(state: GraphState) -> GraphState:
    """Web scrape to retrieve article text."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    potential_articles = []
    for article in state["articles_metadata"]:
        if article['url'] not in state["scraped_urls"]:
            try:
                response = requests.get(article['url'], headers=headers, timeout=10)
                if response.ok:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = soup.get_text(strip=True, separator=' ')
                    potential_articles.append({
                        "title": article["title"],
                        "url": article["url"],
                        "description": article["description"],
                        "text": text[:10000]  # Truncate for token limits
                    })
                    state["scraped_urls"].append(article['url'])
            except Exception as e:
                print(f"Scraping error: {e}")
    
    state["potential_articles"].extend(potential_articles)
    return state

class SelectedUrls(BaseModel):
    urls: List[str] = Field(description="List of selected URLs")

def select_top_urls(state: GraphState) -> GraphState:
    """Select most relevant articles."""
    try:
        prompt = (
            f"Based on the user news query: {state['news_query']}\n\n"
            "Here are the available articles:\n"
            f"{[f'{i+1}. Title: {article['title']}\nURL: {article['url']}\nDescription: {article['description']}\n' for i, article in enumerate(state['potential_articles'])]}\n\n"
            f"Please select exactly {state['num_articles_tldr']} most relevant articles.\n"
            "Respond with ONLY the complete URLs, one per line, no other text."
        )
        
        print("Selecting top URLs...")
        result = llm.invoke(prompt).content
        
        # Clean up URLs - remove any trailing '>' and whitespace
        urls = [url.strip('> \n') for url in re.findall(r'https?://[^\s<>]+', result)]
        
        print(f"Selected URLs: {urls}")
        
        # Match URLs exactly with potential articles
        state["tldr_articles"] = [
            article for article in state["potential_articles"] 
            if any(article['url'].strip() == url.strip() for url in urls)
        ][:state["num_articles_tldr"]]
        
        # Fallback if no matches found
        if not state["tldr_articles"] and state["potential_articles"]:
            print("No URL matches found, using fallback selection")
            state["tldr_articles"] = state["potential_articles"][:state["num_articles_tldr"]]
        
        print(f"Selected {len(state['tldr_articles'])} articles for TL;DR")
        
    except Exception as e:
        print(f"Error selecting top URLs: {e}")
        # Fallback to first n articles if selection fails
        state["tldr_articles"] = state["potential_articles"][:state["num_articles_tldr"]]
    
    return state

async def summarize_articles_parallel(state: GraphState) -> GraphState:
    """Generate summaries concurrently."""
    prompt_template = (
        "Create bullet points for: {text}\n"
        "Format as:\n"
        "{title}\n"
        "{url}\n"
        "* Key point 1\n"
        "* Key point 2\n"
        "* Key point 3"
    )
    
    async def summarize(article):
        try:
            result = await llm.ainvoke(prompt_template.format(
                title=article["title"],
                url=article["url"],
                text=article["text"][:5000]  # Truncate for token limits
            ))
            return result.content
        except Exception as e:
            print(f"Summary error: {e}")
            return "Summary unavailable"
    
    summaries = await asyncio.gather(*[
        summarize(a) for a in state["tldr_articles"]
    ])
    
    for article, summary in zip(state["tldr_articles"], summaries):
        article["summary"] = summary
    
    return state

# Rest of the code remains the same for workflow setup and execution

def create_workflow():
    workflow = Graph()

    workflow.set_entry_point("generate_tavily_params")

    # Add nodes
    workflow.add_node("generate_tavily_params", generate_tavily_params)
    workflow.add_node("retrieve_articles_metadata", retrieve_articles_metadata)
    workflow.add_node("retrieve_articles_text", retrieve_articles_text)
    workflow.add_node("select_top_urls", select_top_urls)
    workflow.add_node("summarize_articles", summarize_articles_parallel)  # Changed this line
    workflow.add_node("format_results", format_results)

    # Add edges
    workflow.add_edge("generate_tavily_params", "retrieve_articles_metadata")
    workflow.add_edge("retrieve_articles_metadata", "retrieve_articles_text")

    workflow.add_conditional_edges(
        "retrieve_articles_text",
        articles_text_decision,
        {
            "generate_tavily_params": "generate_tavily_params",
            "select_top_urls": "select_top_urls",
            "END": END
        }
    )

    workflow.add_edge("select_top_urls", "summarize_articles")
    workflow.add_edge("summarize_articles", "format_results")  # Changed this line
    workflow.add_edge("format_results", END)

    return workflow.compile()

# Run Function
async def run_workflow(query: str, num_searches_remaining: int = 3, num_articles_tldr: int = 3):
    """Run the news summarization workflow."""
    initial_state = {
        "news_query": query,
        "num_searches_remaining": num_searches_remaining,
        "search_params": {},
        "past_searches": [],
        "articles_metadata": [],
        "scraped_urls": [],
        "num_articles_tldr": num_articles_tldr,
        "potential_articles": [],
        "tldr_articles": [],
        "formatted_results": ""
    }
    
    try:
        app = create_workflow()
        result = await app.ainvoke(initial_state)
        if not result["formatted_results"]:
            return "No results found. Please try a different search query."
        return result["formatted_results"]
    except Exception as e:
        import traceback
        print(f"Workflow error: {str(e)}")
        print(traceback.format_exc())
        return f"An error occurred while processing your request: {str(e)}"
async def main():
    query = "what are the top genai news of today?"
    result = await run_workflow(query, num_searches_remaining=2, num_articles_tldr=3)
    print(result)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())