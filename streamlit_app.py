import streamlit as st
import pandas as pd
import joblib
import os

# Page Configuration
st.set_page_config(page_title="TED Talks Recommendation (Deep Learning)", page_icon="🧠", layout="wide")

st.title("🧠 TED Talks Recommendation System (Deep Learning)")
st.markdown("""
This system analyzes the semantic content of talks using **Sentence-BERT (Transformer)** models and provides recommendations.
""")

# ----------------------------------------------------------------
# MODEL YÜKLEME (PATH FIX)
# ----------------------------------------------------------------
@st.cache_resource
def load_assets():
    # 1. Bu script'in (streamlit_app.py) tam konumunu bul
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Dosya yollarını bu konuma göre oluştur (Hata şansını sıfırlar)
    model_path = os.path.join(current_dir, 'models', 'cosine_sim_dl.pkl')
    data_path = os.path.join(current_dir, 'models', 'ted_data.pkl')
    indices_path = os.path.join(current_dir, 'models', 'indices.pkl')
    
    # 3. Yükleme İşlemi
    try:
        # Önce dosyalar orada mı diye kontrol edelim (Debug için)
        if not os.path.exists(model_path):
            st.error(f"Dosya bulunamadı: {model_path}")
            return None, None, None

        cosine_sim = joblib.load(model_path)
        df = joblib.load(data_path)
        indices = joblib.load(indices_path)
        return cosine_sim, df, indices

    except Exception as e:
        st.error(f"Files could not be loaded: {e}")
        st.info("Ensure 'cosine_sim_dl.pkl', 'ted_data.pkl', and 'indices.pkl' are in the 'src/' folder.")
        return None, None, None

cosine_sim, df, indices = load_assets()

# ----------------------------------------------------------------
# ARAYÜZ (INTERFACE)
# ----------------------------------------------------------------
if df is not None:
    st.sidebar.header("Settings")
    num_recommendations = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
    
    selected_talk = st.selectbox("Select a Talk:", df['title'].values)
    
    if st.button("Recommend with Deep Learning"):
        try:
            # Seçilen konuşmanın indeksini al
            idx = indices[selected_talk]
            
            # Benzerlik skorlarını al
            sim_scores = list(enumerate(cosine_sim[idx]))
            
            # Sırala (En yüksek skor en üstte)
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # İlk N tanesini al (Kendisi hariç, o yüzden 1'den başlıyoruz)
            sim_scores = sim_scores[1:num_recommendations+1]
            
            # İndeksleri eşleştir
            talk_indices = [i[0] for i in sim_scores]
            recommendations = df.iloc[talk_indices]
            
            st.subheader(f"Semantically Most Similar Talks to '{selected_talk}':")
            
            for i, row in recommendations.iterrows():
                with st.expander(f"{row['title']} - {row['main_speaker']}"):
                    st.write(f"**Speaker:** {row['main_speaker']}")
                    st.write(f"[Watch]({row['url']})")
                    
        except KeyError:
            st.error("Selected talk not found in the index.")
        except Exception as e:
            st.error(f"An error occurred during recommendation: {e}")
else:
    # Model yüklenemediyse uyarı ver
    st.warning("Waiting for model files (DL). Please ensure .pkl files are in 'src/' folder.")