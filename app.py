from flask import Flask, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('cosmetics.csv')  # Ensure the path is correct for your CSV

# For simplicity, I am assuming the dataframe structure based on your CSV.
df['Liked'] = np.random.choice([0, 1], size=len(df))  # Simulate user feedback for testing
df['features'] = df['Ingredients']  # You can change this to another feature if necessary

# Create a TF-IDF vectorizer and fit on the 'features' column (Ingredients or any other text-based feature)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['features'])

# Compute cosine similarity between all products
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a series to map product names to their indices
indices = pd.Series(df.index, index=df['Name']).drop_duplicates()

# Function to recommend products based on input filters
def recommend_cosmetics(skin_type, label_filter, rank_filter, brand_filter, price_range, ingredient_input=None, num_recommendations=10):
    # Filter products based on skin type and other filters first
    recommended_products = df[df[skin_type] == 1]
    
    if label_filter != 'All':
        recommended_products = recommended_products[recommended_products['Label'] == label_filter]
    
    recommended_products = recommended_products[ 
        (recommended_products['Rank'] >= rank_filter[0]) & 
        (recommended_products['Rank'] <= rank_filter[1])
    ]
    
    if brand_filter != 'All':
        recommended_products = recommended_products[recommended_products['Brand'] == brand_filter]
    
    recommended_products = recommended_products[
        (recommended_products['Price'] >= price_range[0]) & 
        (recommended_products['Price'] <= price_range[1])
    ]

    # If ingredient input is provided, recommend products based on ingredient similarity
    if ingredient_input:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['Ingredients'])
        input_vec = vectorizer.transform([ingredient_input])
        cosine_similarities = cosine_similarity(input_vec, tfidf_matrix).flatten()
        recommended_indices = cosine_similarities.argsort()[-num_recommendations:][::-1]
        ingredient_recommendations = df.iloc[recommended_indices]
        recommended_products = recommended_products[recommended_products.index.isin(ingredient_recommendations.index)]
    
    return recommended_products.sort_values(by=['Rank']).head(num_recommendations)

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        # Get input from the form
        skin_type = request.form['skin_type']
        label_filter = request.form['label']
        rank_filter = (float(request.form['rank_range']), 5.0)  # Assuming rank slider gives max value as upper bound
        brand_filter = request.form['brand']
        price_range = (float(request.form['price_range']), 300.0)  # Assuming price slider gives upper bound
        ingredient_input = request.form.get('ingredients', '').strip()

        # Get recommendations
        top_recommended_products = recommend_cosmetics(
            skin_type, label_filter, rank_filter, brand_filter, price_range, ingredient_input
        )

        # Pass the results to the template
        return render_template(
            'recommendation_results.html', recommended_products=top_recommended_products.to_dict(orient='records')
        )
    return render_template('recommendation.html')



@app.route('/shop')
def shop():
    return render_template('shop.html')

@app.route('/checkout')
def checkout():
    return render_template('checkout.html')

@app.route('/cart')
def cart():
    return render_template('cart.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/thankyou')
def thankyou():
    return render_template('thankyou.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')



if __name__ == '__main__':
    app.run(debug=True)

