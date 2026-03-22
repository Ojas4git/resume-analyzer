import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.title("AI Resume Analyzer")

st.header("Find the best job role based on your skills")
st.write("Upload your skills and get job recommendations instantly 🚀")
st.write("---")
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
data = pd.read_csv("jobs.csv")

st.title("AI Resume Analyzer")

# Input
resume = st.text_area("Enter your resume skills:", 
                      placeholder="e.g. python machine learning data analysis")

if resume:
    texts = data["skills"].tolist()
    texts.append(resume)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)

    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    top_indices = np.argsort(similarity[0])[::-1][:2]

    st.subheader("🎯 Recommended Roles:")

    for i in top_indices:
        st.success(data.iloc[i]["role"])

st.write("---")
st.write("Built with ❤️ using Machine Learning and NLP")