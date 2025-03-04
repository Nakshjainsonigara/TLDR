from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import asyncio
from agent import run_workflow

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
async def search():
    try:
        data = request.json
        query = data.get('query', '')
        num_searches = data.get('num_searches', 2)
        num_articles = data.get('num_articles', 3)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400

        result = await run_workflow(
            query=query,
            num_searches_remaining=num_searches,
            num_articles_tldr=num_articles
        )
        
        return jsonify({'result': result})
    except Exception as e:
        print(f"Error in search endpoint: {str(e)}")  # Add logging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 