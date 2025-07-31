from flask import Flask, render_template, jsonify, request
import os

from chuck_norris_api import ChuckNorrisClient

# Get the directory where this script is located
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, template_folder=os.path.join(basedir, 'templates'))
client = ChuckNorrisClient()


@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/api/random-joke')
def random_joke():
    """Get a random joke."""
    category = request.args.get('category')
    try:
        joke, joke_id, elapsed = client.random_joke(category)
        return jsonify({
            'success': True,
            'joke': joke,
            'id': joke_id,
            'elapsed': elapsed,
            'category': category
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/categories')
def categories():
    """Get all available categories."""
    try:
        cats, elapsed = client.categories()
        return jsonify({
            'success': True,
            'categories': cats,
            'elapsed': elapsed
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/search')
def search():
    """Search for jokes."""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))

    if not query:
        return jsonify({
            'success': False,
            'error': 'Query parameter "q" is required'
        }), 400

    try:
        results, elapsed = client.search(query, limit)
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results),
            'query': query,
            'elapsed': elapsed
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/joke/<joke_id>')
def get_joke_by_id(joke_id):
    """Get a specific joke by ID."""
    try:
        joke, elapsed = client.joke_by_id(joke_id)
        return jsonify({
            'success': True,
            'joke': joke,
            'id': joke_id,
            'elapsed': elapsed
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
