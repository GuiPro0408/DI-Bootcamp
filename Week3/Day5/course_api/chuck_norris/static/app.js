/**
 * Chuck Norris Jokes Web Application
 * Main JavaScript functionality for handling API calls and UI interactions
 */

class ChuckNorrisApp {
    constructor() {
        this.totalJokes = 0;
        this.totalTime = 0;
        this.categoriesCount = 0;
        this.init();
    }

    /**
     * Initialize the application
     */
    init() {
        this.bindEvents();
        this.loadCategories().catch(error => console.error('Error loading categories:', error))
    }

    /**
     * Bind event listeners to UI elements
     */
    bindEvents() {
        const getJokeBtn = document.getElementById('getJokeBtn');
        const searchBtn = document.getElementById('searchBtn');
        const searchInput = document.getElementById('searchInput');

        if (getJokeBtn) {
            getJokeBtn.addEventListener('click', () => this.getRandomJoke());
        }

        if (searchBtn) {
            searchBtn.addEventListener('click', () => this.searchJokes());
        }

        if (searchInput) {
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.searchJokes().catch(error => console.error('Error searching jokes:', error))
                }
            });
        }
    }

    /**
     * Load available joke categories from the API
     */
    async loadCategories() {
        try {
            const response = await fetch('/api/categories');
            /** @type {{success: boolean, categories: string[], elapsed: number, error?: string}} */
            const data = await response.json();

            if (data.success && data.categories) {
                this.populateCategorySelect(data.categories);
                this.categoriesCount = data.categories.length;
                this.updateStats();
            } else {
                this.showError('Failed to load categories: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Error loading categories:', error);
            this.showError('Failed to load categories');
        }
    }

    /**
     * Populate the category select dropdown
     * @param {string[]} categories - Array of category strings
     */
    populateCategorySelect(categories) {
        const select = document.getElementById('categorySelect');
        if (!select) return;

        // Clear existing options except the first one
        while (select.children.length > 1) {
            select.removeChild(select.lastChild);
        }

        categories.forEach(category => {
            const option = document.createElement('option');
            option.value = category;
            option.textContent = category.charAt(0).toUpperCase() + category.slice(1);
            select.appendChild(option);
        });
    }

    /**
     * Get a random joke from the API
     */
    async getRandomJoke() {
        const categorySelect = document.getElementById('categorySelect');
        const btn = document.getElementById('getJokeBtn');

        if (!btn) return;

        const category = categorySelect ? categorySelect.value : '';

        this.setButtonLoading(btn, 'Loading...');
        this.hideError();
        this.hideSearchResults();

        try {
            const url = category ? `/api/random-joke?category=${encodeURIComponent(category)}` : '/api/random-joke';
            const response = await fetch(url);
            const data = await response.json();

            if (data.success) {
                this.displayJoke(data);
                this.updateStats(data.elapsed);
            } else {
                this.showError(data.error || 'Failed to fetch joke');
            }
        } catch (error) {
            console.error('Error fetching joke:', error);
            this.showError('Failed to fetch joke. Please try again.');
        } finally {
            this.resetButton(btn, 'Get Random Joke');
        }
    }

    /**
     * Search for jokes based on user input
     */
    async searchJokes() {
        const searchInput = document.getElementById('searchInput');
        const btn = document.getElementById('searchBtn');

        if (!searchInput || !btn) return;

        const query = searchInput.value.trim();
        if (!query) {
            this.showError('Please enter a search term');
            return;
        }

        this.setButtonLoading(btn, 'Searching...');
        this.hideError();

        try {
            const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&limit=10`);
            const data = await response.json();

            if (data.success) {
                this.displaySearchResults(data);
                this.updateStats(data.elapsed);
            } else {
                this.showError(data.error || 'Failed to search jokes');
            }
        } catch (error) {
            console.error('Error searching jokes:', error);
            this.showError('Failed to search jokes. Please try again.');
        } finally {
            this.resetButton(btn, 'Search');
        }
    }

    /**
     * Display a joke in the main joke display area
     * @param {Object} data - Joke data object
     */
    displayJoke(data) {
        const jokeText = document.getElementById('jokeText');
        const jokeId = document.getElementById('jokeId');
        const responseTime = document.getElementById('responseTime');
        const jokeCategory = document.getElementById('jokeCategory');
        const jokeMeta = document.getElementById('jokeMeta');
        const jokeDisplay = document.getElementById('jokeDisplay');

        if (jokeText) {
            jokeText.textContent = data.joke;
        }

        if (jokeId) {
            jokeId.textContent = data.id;
        }

        if (responseTime) {
            responseTime.textContent = String(Math.round(data.elapsed * 1000));
        }

        if (jokeCategory) {
            jokeCategory.textContent = data.category || 'Random';
        }

        if (jokeMeta) {
            jokeMeta.classList.remove('hidden');
        }

        if (jokeDisplay) {
            jokeDisplay.classList.add('fade-in');
            setTimeout(() => {
                jokeDisplay.classList.remove('fade-in');
            }, 500);
        }
    }

    /**
     * Display search results
     * @param {Object} data - Search results data
     */
    displaySearchResults(data) {
        const container = document.getElementById('resultsContainer');
        const searchResults = document.getElementById('searchResults');

        if (!container || !searchResults) return;

        container.innerHTML = '';

        if (data.results.length === 0) {
            container.innerHTML = '<p>No jokes found for your search term.</p>';
        } else {
            data.results.forEach(result => {
                const div = document.createElement('div');
                div.className = 'search-result';
                div.textContent = result.value;
                div.addEventListener('click', () => {
                    this.displayJoke({
                        joke: result.value,
                        id: result.id,
                        elapsed: 0,
                        category: result.categories ? result.categories[0] : 'Unknown'
                    });
                    this.hideSearchResults();
                });
                container.appendChild(div);
            });
        }

        searchResults.classList.remove('hidden');
        searchResults.classList.add('fade-in');

        setTimeout(() => {
            searchResults.classList.remove('fade-in');
        }, 500);
    }

    /**
     * Update statistics display
     * @param {number|null} responseTime - Response time in seconds
     */
    updateStats(responseTime = null) {
        if (responseTime !== null) {
            this.totalJokes++;
            this.totalTime += responseTime;
        }

        const totalJokesEl = document.getElementById('totalJokes');
        const avgTimeEl = document.getElementById('avgTime');
        const categoriesEl = document.getElementById('categories');

        if (totalJokesEl) {
            totalJokesEl.textContent = String(this.totalJokes);
        }

        if (avgTimeEl) {
            avgTimeEl.textContent = this.totalJokes > 0
                ? Math.round(this.totalTime / this.totalJokes * 1000) + 'ms'
                : '0ms';
        }

        if (categoriesEl) {
            categoriesEl.textContent = String(this.categoriesCount);
        }
    }

    /**
     * Show error message to user
     * @param {string} message - Error message to display
     */
    showError(message) {
        const errorDiv = document.getElementById('errorDisplay');
        if (errorDiv) {
            errorDiv.textContent = message;
            errorDiv.classList.remove('hidden');
        }
    }

    /**
     * Hide error message
     */
    hideError() {
        const errorDiv = document.getElementById('errorDisplay');
        if (errorDiv) {
            errorDiv.classList.add('hidden');
        }
    }

    /**
     * Hide search results
     */
    hideSearchResults() {
        const searchResults = document.getElementById('searchResults');
        if (searchResults) {
            searchResults.classList.add('hidden');
        }
    }

    /**
     * Set button to loading state
     * @param {HTMLElement} button - Button element
     * @param {string} loadingText - Text to display while loading
     */
    setButtonLoading(button, loadingText) {
        button.textContent = loadingText;
        button.disabled = true;
    }

    /**
     * Reset button to normal state
     * @param {HTMLElement} button - Button element
     * @param {string} normalText - Normal button text
     */
    resetButton(button, normalText) {
        button.textContent = normalText;
        button.disabled = false;
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChuckNorrisApp();
});
