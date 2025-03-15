import streamlit as st
from utils import extract_reviews, make_review_df, get_emotions, calculate_sentiment_percentages, create_wordcloud
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

st.title("📊 Sentiment Analysis")
st.write("Analyze chat sentiments, most used words, emojis, and generate word clouds using BERT!")

url = st.text_input('The URL link')
st.write("Enter the URL of the Yelp page you want to analyze.")
st.write("Check out the Sample Data button to see an example of the data that will be analyzed.")
button = st.button("Use Sample Data")

if url:
    reviews = extract_reviews(url) #https://www.yelp.com/biz/social-brew-cafe-pyrmont
    
    if reviews:
        # Create tabs for different analyses
        st.tabs(["Sentiment Analysis"])
        st.subheader("Sentiment Analysis")
        df = make_review_df(reviews)
        st.dataframe(df)

        # Calculate sentiment percentages
        percentages = calculate_sentiment_percentages(df)
        st.write("Sentiment Percentages")
        st.dataframe(percentages)
            
        # Visualize sentiment distribution
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment Score', 'Count']
            
        fig = px.bar(sentiment_counts, x='Sentiment Score', y='Count',
                    labels={'Sentiment Score': 'Sentiment (1-5)', 'Count': 'Number of Reviews'},
                    title='Distribution of Sentiment Scores',
                    color='Sentiment Score',
                    color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig)

        st.subheader("Emotion Analysis")
            
        with st.spinner("Analyzing emotions in reviews..."):
            emotions_df = get_emotions(reviews)
                
            # Display the emotion scores
            combined_df = pd.concat([df, emotions_df], axis=1)
            st.dataframe(combined_df)
                
            # Calculate average emotion scores across all reviews
            emotion_cols = [col for col in emotions_df.columns if col != 'text']
            avg_emotions = emotions_df[emotion_cols].mean().reset_index()
            avg_emotions.columns = ['Emotion', 'Average Score']
                
            # Create bar chart for average emotions
            fig = px.bar(avg_emotions, x='Emotion', y='Average Score',
                        title='Average Emotion Scores Across All Reviews',
                        color='Average Score',
                        color_continuous_scale=px.colors.sequential.Plasma)
            st.plotly_chart(fig)
                
            # Create heatmap for individual reviews and their emotions
            st.subheader("Emotion Heatmap")
                
            # Limit to first 10 reviews if there are many
            display_limit = min(10, len(emotions_df))
            heatmap_data = emotions_df.iloc[:display_limit].copy()
                
            # Create a numeric index for the heatmap
            heatmap_data['Review #'] = [f"Review {i+1}" for i in range(len(heatmap_data))]
                
            # Melt the dataframe for plotly
            melted_df = pd.melt(
                    heatmap_data, 
                    id_vars=['Review #'], 
                    value_vars=emotion_cols,
                    var_name='Emotion', 
                    value_name='Score'
                )
                
                # Create heatmap
            fig = px.density_heatmap(
                    melted_df, 
                    x='Emotion', 
                    y='Review #',
                    z='Score',
                    title='Emotion Intensity by Review',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
            st.plotly_chart(fig)

        st.subheader("Word Cloud")
        wordcloud = create_wordcloud(reviews)
        st.image(wordcloud.to_array(), use_container_width=True)

elif button:
    reviews = extract_reviews("https://www.yelp.com/biz/starbucks-san-francisco-166")

    if reviews:
        # Create tabs for different analyses
        st.tabs(["Sentiment Analysis"])
        st.subheader("Sentiment Analysis")
        df = make_review_df(reviews)
        st.dataframe(df)

        # Calculate sentiment percentages
        percentages = calculate_sentiment_percentages(df)
        st.write("Sentiment Percentages")
        st.dataframe(percentages)
            
        # Visualize sentiment distribution
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment Score', 'Count']
            
        fig = px.bar(sentiment_counts, x='Sentiment Score', y='Count',
                    labels={'Sentiment Score': 'Sentiment (1-5)', 'Count': 'Number of Reviews'},
                    title='Distribution of Sentiment Scores',
                    color='Sentiment Score',
                    color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig)

        st.subheader("Emotion Analysis")
            
        with st.spinner("Analyzing emotions in reviews..."):
            emotions_df = get_emotions(reviews)
                
            # Display the emotion scores
            combined_df = pd.concat([df, emotions_df], axis=1)
            st.dataframe(combined_df)
                
            # Calculate average emotion scores across all reviews
            emotion_cols = [col for col in emotions_df.columns if col != 'text']
            avg_emotions = emotions_df[emotion_cols].mean().reset_index()
            avg_emotions.columns = ['Emotion', 'Average Score']
                
            # Create bar chart for average emotions
            fig = px.bar(avg_emotions, x='Emotion', y='Average Score',
                        title='Average Emotion Scores Across All Reviews',
                        color='Average Score',
                        color_continuous_scale=px.colors.sequential.Plasma)
            st.plotly_chart(fig)
                
            # Create heatmap for individual reviews and their emotions
            st.subheader("Emotion Heatmap")
                
            # Limit to first 10 reviews if there are many
            display_limit = min(10, len(emotions_df))
            heatmap_data = emotions_df.iloc[:display_limit].copy()
                
            # Create a numeric index for the heatmap
            heatmap_data['Review #'] = [f"Review {i+1}" for i in range(len(heatmap_data))]
                
            # Melt the dataframe for plotly
            melted_df = pd.melt(
                    heatmap_data, 
                    id_vars=['Review #'], 
                    value_vars=emotion_cols,
                    var_name='Emotion', 
                    value_name='Score'
                )
                
                # Create heatmap
            fig = px.density_heatmap(
                    melted_df, 
                    x='Emotion', 
                    y='Review #',
                    z='Score',
                    title='Emotion Intensity by Review',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
            st.plotly_chart(fig)

        st.subheader("Word Cloud")
        wordcloud = create_wordcloud(reviews)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.pyplot()
        


# st.write(df)
# st.write(most_used_words)
