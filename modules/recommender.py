import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def get_basket_data(df, country="France"):
    """
    Prepares the data for Market Basket Analysis by filtering a specific country
    and creating a basket matrix (Invoice vs Product).
    Using 'France' or 'Germany' is recommended for faster processing than 'UK'.
    """
    # Filter by Country
    basket = (df[df['Country'] == country]
              .groupby(['Invoice', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('Invoice'))
    
    # Convert quantities to 0/1 (Boolean)
    def encode_units(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1

    basket_sets = basket.applymap(encode_units)
    return basket_sets

def generate_rules(basket_sets, min_support=0.05, min_confidence=0.2):
    """
    Runs the Apriori algorithm to find frequent itemsets and association rules.
    """
    # 1. Find Frequent Itemsets
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    
    if frequent_itemsets.empty:
        return pd.DataFrame() # Return empty if no patterns found
    
    # 2. Generate Rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    
    # Filter by confidence
    rules = rules[rules['confidence'] >= min_confidence]
    
    return rules

def recommend_products(rules, product_name):
    """
    Returns a list of recommended products based on the rules for a specific item.
    """
    # Filter rules where the antecedent (if part) contains the product_name
    recommendations = rules[rules['antecedents'].apply(lambda x: product_name in x)]
    
    # Sort by confidence (likelihood)
    recommendations = recommendations.sort_values('confidence', ascending=False)
    
    return recommendations[['consequents', 'confidence', 'lift']]