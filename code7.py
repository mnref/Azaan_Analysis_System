import streamlit as st
import re
import tempfile
from pydub import AudioSegment
from pdf_export import create_tafsili_report_pdf, add_download_button  # Save the above code as pdf_export.py
from google.oauth2 import service_account
from google.cloud import speech, storage
import openai
import librosa
import librosa.display
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from io import BytesIO
from fuzzywuzzy import fuzz
from scipy.stats import pearsonr
import json

# Set OpenAI API key
openai.api_key = st.secrets["openai"] # Replace with your actual OpenAI API Key


# Set Google Cloud Service Account credentials
google_creds = st.secrets["google"]

# Initialize the Google Cloud clients with the credentials
credentials = service_account.Credentials.from_service_account_info(google_creds)
speech_client = speech.SpeechClient(credentials=credentials)
storage_client = storage.Client(credentials=credentials)

# Predefined data
predefined_spectrograms = {
    'Muazzin Imran': cv2.imread("Qari Imran.jpg"),
    'Muazzin Mishary': cv2.imread("Mufti Mishary.jpg"),
    'Muazzin Mufti Menk': cv2.imread("Mufti Menk.jpg"),
}

expert_audio_paths = {
    'Muazzin Imran': "Zohar macca.mp3",
    'Muazzin Mishary': "zohar by sheik mishary.mp3",
    'Muazzin Mufti Menk': "Zohar.mp3",
}

expert_transcriptions = {
    'Muazzin Imran': """الله أكبر الله أكبر الله أكبر الله أكبر أشهد أن لا إله إلا الله أشهد أن لا إله إلا الله أشهد أن محمدا رسول الله أشهد أن محمدا رسول الله حي على الصلاة حي على الصلاة حي على الفلاح حي على الفلاح الله أكبر الله أكبر لا إله إلا الله""",
    'Muazzin Mishary': """الله أكبر الله أكبر الله أكبر الله أكبر أشهد أن لا إله إلا الله أشهد أن لا إله إلا الله أشهد أن محمدا رسول الله أشهد أن محمدا رسول الله حي على الصلاة حي على الصلاة حي على الفلاح حي على الفلاح الله أكبر الله أكبر لا إله إلا الله""",
    'Muazzin Mufti Menk': """الله أكبر الله أكبر الله أكبر الله أكبر أشهد أن لا إله إلا الله أشهد أن لا إله إلا الله أشهد أن محمدا رسول الله أشهد أن محمدا رسول الله حي على الصلاة حي على الصلاة حي على الفلاح حي على الفلاح الله أكبر الله أكبر لا إله إلا الله"""
}

# Audio Feature Extraction and Analysis Functions
def extract_audio_features(audio_path):
    """Extract relevant audio features for analysis"""
    y, sr = librosa.load(audio_path)
    
    # Extract features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Calculate statistics
    pitch_stats = {
        'mean': np.mean(pitches[magnitudes > np.max(magnitudes) * 0.1]),
        'std': np.std(pitches[magnitudes > np.max(magnitudes) * 0.1]),
        'range': np.ptp(pitches[magnitudes > np.max(magnitudes) * 0.1])
    }
    
    return {
        'tempo': float(tempo),
        'pitch_stats': pitch_stats,
        'dynamics': {
            'mean_rms': float(np.mean(rms)),
            'max_rms': float(np.max(rms)),
            'dynamic_range': float(np.ptp(rms))
        },
        'articulation': float(np.mean(zero_crossings)),
        'timbre': {
            'mfcc_means': [float(np.mean(mfcc)) for mfcc in mfccs],
            'mfcc_stds': [float(np.std(mfcc)) for mfcc in mfccs]
        }
    }

def display_pronunciation_analysis(diagnosis_result):
    st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
            <h3 style="color: #234e70; margin-bottom: 1rem;">Talaffuz ka Jaiza (Pronunciation Analysis)</h3>
            <div style="border-left: 4px solid #1ca4a4; padding-left: 1rem; margin: 1rem 0;">
    """, unsafe_allow_html=True)
    
    if 'pronunciation_analysis' in diagnosis_result:
        for issue in diagnosis_result['pronunciation_analysis']:
            st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <span style="color: #1ca4a4;">Word {issue['word_number']}</span>: 
                    <span style="color: #d4af37;">{issue['user_word']}</span> → 
                    <span style="color: #234e70;">{issue['correct_word']}</span>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def display_voice_quality_analysis(diagnosis_result):
    st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
            <h3 style="color: #234e70; margin-bottom: 1rem;">Awaaz ki Kaifiyat (Voice Quality Analysis)</h3>
            <div style="border-left: 4px solid #1ca4a4; padding-left: 1rem; margin: 1rem 0;">
    """, unsafe_allow_html=True)
    
    if 'audio_analysis' in diagnosis_result:
        metrics = {
            'Taal/Laye (Tempo)': diagnosis_result['audio_analysis']['tempo_diff'],
            'Pitch Accuracy': diagnosis_result['audio_analysis']['pitch_diff']['mean'],
            'Voice Control': diagnosis_result['audio_analysis']['dynamics_diff']['mean_rms'],
            'Overall Similarity': diagnosis_result['audio_analysis']['timbre_similarity']
        }
        
        for metric, value in metrics.items():
            quality = "Excellent" if value < 0.3 else "Good" if value < 0.6 else "Needs Improvement"
            color = "#1ca4a4" if value < 0.3 else "#d4af37" if value < 0.6 else "#dc3545"
            
            st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <strong>{metric}</strong>: 
                    <span style="color: {color};">{quality}</span>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
def display_detailed_report(diagnosis_result):
    st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); margin-top: 2rem;">
            <h3 style="color: #234e70; margin-bottom: 1rem; text-align: center;">
                <span style="border-bottom: 3px solid #1ca4a4; padding-bottom: 5px;">
                    Tafsili Report (Detailed Analysis)
                </span>
            </h3>
    """, unsafe_allow_html=True)
    
    if 'detailed_report' in diagnosis_result:
        # Display report content
        sections = diagnosis_result['detailed_report'].split('\n\n')
        for section in sections:
            if section.strip():
                if any(header in section.lower() for header in ['talaffuz', 'pronunciation', 'awaaz', 'voice', 'observations', 'mushahidaat', 'pattern', 'khaas nukaat']):
                    header_html = """
                        <div style="margin: 1.5rem 0 1rem 0;">
                            <h4 style="color: #1ca4a4; font-size: 1.2em; margin-bottom: 0.5rem;">
                                {0}
                            </h4>
                            <div style="background: #1ca4a4; height: 2px; width: 50px; margin: 0.5rem 0;"></div>
                            <div style="border-left: 4px solid #1ca4a4; padding-left: 1rem; margin-top: 1rem;">
                                {1}
                            </div>
                        </div>
                    """
                    header_content = section.split('\n')[0]
                    remaining_content = section.split('\n', 1)[1].replace('\n', '<br>') if '\n' in section else ''
                    st.markdown(header_html.format(header_content, remaining_content), unsafe_allow_html=True)
                else:
                    content_html = """
                        <div style="margin: 1rem 0; padding-left: 1rem;">
                            {0}
                        </div>
                    """
                    st.markdown(content_html.format(section.replace('\n', '<br>')), unsafe_allow_html=True)
        
        # Add download button right after the report content
        try:
            pdf_buffer = create_tafsili_report_pdf(diagnosis_result['detailed_report'])
            st.download_button(
                label="📥 Download Tafsili Report (PDF)",
                data=pdf_buffer,
                file_name="tafsili_report.pdf",
                mime="application/pdf",
                key="download_report"
            )
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)
def compare_audio_features(user_features, expert_features):
    """Compare user's audio features with expert's features"""
    differences = {
        'tempo_diff': abs(user_features['tempo'] - expert_features['tempo']),
        'pitch_diff': {
            'mean': abs(user_features['pitch_stats']['mean'] - expert_features['pitch_stats']['mean']),
            'std': abs(user_features['pitch_stats']['std'] - expert_features['pitch_stats']['std'])
        },
        'dynamics_diff': {
            'mean_rms': abs(user_features['dynamics']['mean_rms'] - expert_features['dynamics']['mean_rms']),
            'dynamic_range': abs(user_features['dynamics']['dynamic_range'] - expert_features['dynamics']['dynamic_range'])
        },
        'articulation_diff': abs(user_features['articulation'] - expert_features['articulation']),
        'timbre_similarity': pearsonr(
            user_features['timbre']['mfcc_means'],
            expert_features['timbre']['mfcc_means']
        )[0]
    }
    return differences

def analyze_pronunciation(transcription, expert_transcription):
    """Analyze pronunciation differences using text comparison"""
    words_user = transcription.split()
    words_expert = expert_transcription.split()
    
    differences = []
    for i, (word_user, word_expert) in enumerate(zip(words_user, words_expert)):
        if word_user != word_expert:
            differences.append({
                'word_number': i + 1,
                'user_word': word_user,
                'correct_word': word_expert
            })
    
    return differences

def generate_diagnosis_report(audio_comparison, pronunciation_analysis, closest_muezzin):
    """Generate an easy-to-understand diagnosis report in Roman Urdu"""
    
    # Define common issues and their simple explanations
    pronunciation_guide = {
        'الله': {
            'key_aspects': ['Allah lafz ka talaffuz', 'Laam ki gadrahat', 'Ha ki awaaz'],
            'explanation': 'Allah lafz mein Laam ko mota aur Ha ko halki awaaz'
        },
        'أكبر': {
            'key_aspects': ['Akbar mein Kaaf ki awaaz', 'Ber ki awaaz', 'Ra ki awaaz'],
            'explanation': 'Akbar mein Kaaf ko saaf aur Ra ko halka'
        },
        'أشهد': {
            'key_aspects': ['Ash ki awaaz', 'Ha ki awaaz', 'Dal ki awaaz'],
            'explanation': 'Ashhadu mein Ha ko halq se ada karna'
        },
        'حي': {
            'key_aspects': ['Ha ki awaaz', 'Ya ki awaaz'],
            'explanation': 'Hayya mein Ha ko halq se ada karna'
        },
        'على': {
            'key_aspects': ['Ain ki awaaz', 'Laam ki awaaz'],
            'explanation': 'Ala mein Ain ko gehri awaaz'
        }
    }

    prompt = f"""
    As an Azaan expert, provide a simple and clear diagnosis in Roman Urdu that any beginner can understand. Focus on explaining what was observed in the recitation without using technical terms.

    Audio Measurements:
    - Taal/Laye ka farq: {audio_comparison['tempo_diff']:.2f} 
    - Awaaz ki bulandee ka farq: {audio_comparison['pitch_diff']['mean']:.2f}
    - Awaaz ki yaksaniyat: {audio_comparison['dynamics_diff']['mean_rms']:.2f}
    - Talaffuz ki safai: {audio_comparison['articulation_diff']:.2f}
    - Awaaz ki mushababat: {audio_comparison['timbre_similarity']:.2f}

    Talaffuz ka Jaiza:
    {json.dumps(pronunciation_analysis, indent=2, ensure_ascii=False)}

    Please provide a detailed but simple analysis in the following format:

    1. Talaffuz (Pronunciation):
       - Halaq se nikalne wali awaazein (jaise Ha, Ain, Hamza)
       - Zuban se nikalne wali awaazein (jaise Ra, Laam)
       - Honthon se nikalne wali awaazein (jaise Meem, Ba)
       - Ghunna wali awaazein (Noon, Meem ki ghunna)

    2. Awaaz ki Kaifiyat (Voice Quality):
       - Awaaz ki bulandee
       - Awaaz ka utaar chadhao
       - Saansein lene ka tariqa
       - Awaaz ki safai

    3. Khaas Nukaat (Special Points):
       - Allah lafz ka talaffuz
       - Akbar mein Kaaf ki awaaz
       - Hayya mein Ha ki awaaz
       - Assalah aur Alfalah ke alfaaz

    4. Mushahidaat (Observations):
       - Khaas taqat ke pehlu
       - Behtar karne wale pehlu
       - Azaan ki yaksaniyat
       - Awaaz ki khususiyaat

    The response should be in simple Roman Urdu, easy for a common person to understand, and focus on clear observations without using technical Arabic terms.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "user", 
                "content": prompt
            }],
            temperature=0.7,
            max_tokens=1500,
            top_p=1.0
        )
        
        # Get initial analysis
        basic_analysis = response.choices[0].message["content"].strip()
        
        # Add simplified pattern analysis
        pattern_prompt = f"""
        Based on the above analysis, provide a simple explanation in Roman Urdu about:

        1. Azaan ki Tarteeб (Overall Flow):
           - Har jumle ka andaaz
           - Saans lene ke mawaqay
           - Jumlon ka rabt
           - Awaaz ka utaar chadhao

        2. Mukhtlif Hisson ka Jaiza (Different Parts):
           - Shuru ke jumlon ki kaifiyat
           - Darmiyan ke jumlon ki kaifiyat
           - Aakhri jumlon ki kaifiyat

        Use very simple language that a common person can understand easily.
        """
        
        pattern_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "assistant", "content": basic_analysis},
                {"role": "user", "content": pattern_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Combine into user-friendly report
        complete_diagnosis = f"""
Azan ke Talaffuz ka Jaiza
=========================

Bunyadi Jaiza:
-------------
{basic_analysis}

Tafsili Mushahidaat:
------------------
{pattern_response.choices[0].message["content"]}

Tehniki Paimaaish:
----------------
- Taal/Laye: {'Behtar' if audio_comparison['tempo_diff'] < 10 else 'Mazeed Behtar Ho Sakti Hai'}
- Awaaz ki Bulandee: {'Munasib' if audio_comparison['pitch_diff']['mean'] < 50 else 'Thodi Ziada'}
- Awaaz ki Yaksaniyat: {'Achhi' if audio_comparison['dynamics_diff']['mean_rms'] < 0.5 else 'Mazeed Behtar Ho Sakti Hai'}
- Talaffuz ki Safai: {'Waazeh' if audio_comparison['articulation_diff'] < 0.3 else 'Mazeed Waazeh Ho Sakti Hai'}
"""
        return complete_diagnosis
    
    except Exception as e:
        return f"Diagnosis report banane mein masla pesh aaya: {str(e)}"
        
def diagnose_azaan(user_audio_path, expert_audio_path, user_transcription, expert_transcription, closest_muezzin):
    """Main function to generate complete Azaan diagnosis"""
    
    # Extract audio features
    user_features = extract_audio_features(user_audio_path)
    expert_features = extract_audio_features(expert_audio_path)
    
    # Compare audio features
    audio_comparison = compare_audio_features(user_features, expert_features)
    
    # Analyze pronunciation
    pronunciation_analysis = analyze_pronunciation(user_transcription, expert_transcription)
    
    # Generate detailed report
    diagnosis_report = generate_diagnosis_report(audio_comparison, pronunciation_analysis, closest_muezzin)
    
    return {
        'audio_analysis': audio_comparison,
        'pronunciation_analysis': pronunciation_analysis,
        'detailed_report': diagnosis_report
    }

# Original Functions
def upload_audio_to_gcs(file, bucket_name="azaan_bucket"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_audio_file.write(file.getbuffer())
        temp_audio_file.flush()
        
        blob_name = f"demo/testing audios/{temp_audio_file.name.split('/')[-1]}"
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(temp_audio_file.name)
        
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        return gcs_uri

def download_audio_from_gcs(gcs_uri):
    bucket_name = gcs_uri.split("/")[2]
    blob_name = "/".join(gcs_uri.split("/")[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        blob.download_to_filename(temp_audio_file.name)
        return temp_audio_file.name

def transcribe_audio_gcs(gcs_uri):
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=48000,
        language_code="ar"
    )
    operation = speech_client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=180)
    full_transcript = " ".join([result.alternatives[0].transcript for result in response.results])
    return full_transcript.strip()

def normalize_text(text):
    text = re.sub(r'[ًٌٍَُِ]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه")
    return text

def validate_with_openai(transcription):
    correct_azaan = """
    الله أكبر الله أكبر
    الله أكبر الله أكبر
    أشهد أن لا إله إلا الله
    أشهد أن لا إله إلا الله
    أشهد أن محمدًا رسول الله
    أشهد أن محمدًا رسول الله
    حي على الصلاة
    حي على الصلاة
    حي على الفلاح
    حي على الفلاح
    الله أكبر الله أكبر
    لا إله إلا الله
    """

    prompt = f"""
    You are an expert in validating the Azaan (the call to prayer). Below is the correct structure of the Azaan. 
    Compare the transcription provided with this structure to determine if it contains all essential phrases in the correct order.

    Validation Guidelines:
    - Validate the Azaan as "VALIDATED" if it contains all essential phrases in the correct sequence, even if there are minor spelling, diacritic, or punctuation differences.
    - Specifically, ignore small differences such as:
        - Missing or extra diacritics (e.g., "ا" vs. "أ" or "حي على الصلاه" vs. "حي على الصلاة").
        - Minor spelling variations, such as:
            - "لا اله الا الله" vs. "لا إله إلا الله".
            - "حي على الصلاه" vs. "حي على الصلاة".
            - "حي على الفلاح" vs. "حي على الفلاح".
            - "أشهد" vs "شهاده"
        - Punctuation or slight variations in commonly understood words and phrases.
    - Invalidate the Azaan as "INVALIDATED" only if:
        - Essential phrases are missing.
        - Extra, unrelated phrases that are not part of the Azaan are added.
        - Major incorrect words or substitutions that change the meaning of an essential phrase are present.

    Correct Azaan Structure:
    {correct_azaan}

    Transcribed Azaan:
    "{transcription}"

    Conclude with "Validation Status: VALIDATED" if the Azaan matches the correct structure, or "Validation Status: INVALIDATED" if it does not, and list any specific issues if found. Only list issues if they involve missing phrases, extra phrases, or significant meaning changes.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250,
            top_p=1.0,
            stop=None
        )
        validation_result = response.choices[0].message["content"].strip()
        st.write("Response:", validation_result)
        return validation_result
    except Exception as e:
        st.error(f"Error with OpenAI API request: {e}")
        return None

def generate_mel_spectrogram(audio_file):
    y, sr = librosa.load(audio_file, sr=22050)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db, sr

def plot_spectrogram(mel_spectrogram, sr):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mel_spectrogram, sr=sr, x_axis='time', y_axis='mel', cmap='magma', ax=ax)
    ax.set(title="Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    plt.close(fig)
    return buf

def calculate_similarity(image1, image2):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGBA2GRAY) if image1.shape[2] == 4 else cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGBA2GRAY) if image2.shape[2] == 4 else cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    if image1_gray.shape != image2_gray.shape:
        image2_gray = cv2.resize(image2_gray, (image1_gray.shape[1], image1_gray.shape[0]))
    return np.mean((image1_gray - image2_gray) ** 2)

import streamlit as st
from streamlit.components.v1 import html
import base64
from pathlib import Path

def set_page_config():
    st.set_page_config(
        page_title="Azaan Analysis System",
        page_icon="🕌",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

def load_css():
    st.markdown("""
        <style>
        /* Main container styling */
        .stApp {
            background: linear-gradient(135deg, #f4e4bc, #fff);
        }
        
        /* Header styling */
        .main-header {
            text-align: center;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            margin-bottom: 2rem;
        }
        
        /* Custom upload button */
        .stButton>button {
            background-color: #1ca4a4;
            color: white;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #234e70;
            transform: translateY(-2px);
        }
        
        /* Progress bars */
        .stProgress > div > div {
            background-color: #1ca4a4;
        }
        
        /* Cards styling */
        .css-1r6slb0 {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        /* Metrics styling */
        .css-1r6slb0.e16fv1kl3 {
            background: white;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
import arabic_reshaper
from bidi.algorithm import get_display
import os
import io

def setup_arabic_font():
    # Register Arabic font - ensure this font file exists in your project
    font_path = r"C:\Users\USER\Downloads\Amiri-Regular.ttf"  # Update with actual path
    pdfmetrics.registerFont(TTFont('Arabic', font_path))
    
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

def create_tafsili_report_pdf(detailed_report):
    buffer = io.BytesIO()
    
    # Register Amiri font for Arabic text
    amiri_font_path = r"C:\Users\USER\Downloads\Amiri-Regular.ttf"
    pdfmetrics.registerFont(TTFont('Amiri', amiri_font_path))
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='ArabicStyle',
        fontName='Amiri',
        fontSize=12,
        alignment=2,
        leading=16
    ))
    
    story = []
    sections = detailed_report.split('\n')
    
    for section in sections:
        if section.strip():
            # Handle Arabic text
            if any('\u0600' <= c <= '\u06FF' for c in section):
                reshaped_text = arabic_reshaper.reshape(section)
                bidi_text = get_display(reshaped_text)
                p = Paragraph(bidi_text, styles['ArabicStyle'])
            else:
                # Non-Arabic text uses normal style
                p = Paragraph(section, styles['Normal'])
            story.append(p)
            story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def add_download_button(detailed_report):
    # Generate PDF
    pdf_buffer = create_tafsili_report_pdf(detailed_report)
    
    # Create download button
    st.download_button(
        label="📥 Download Tafsili Report (PDF)",
        data=pdf_buffer,
        file_name="tafsili_report.pdf",
        mime="application/pdf",
        help="Download the detailed analysis report in PDF format"
    )

import streamlit as st
import re
import tempfile
from pydub import AudioSegment
from pdf_export import create_tafsili_report_pdf, add_download_button  # Save the above code as pdf_export.py
from google.oauth2 import service_account
from google.cloud import speech, storage
import openai
import librosa
import librosa.display
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from io import BytesIO
from fuzzywuzzy import fuzz
from scipy.stats import pearsonr
import json

# Set OpenAI API key
openai.api_key = st.secrets["openai"] # Replace with your actual OpenAI API Key


# Set Google Cloud Service Account credentials
google_creds = st.secrets["google"]

# Initialize the Google Cloud clients with the credentials
credentials = service_account.Credentials.from_service_account_info(google_creds)
speech_client = speech.SpeechClient(credentials=credentials)
storage_client = storage.Client(credentials=credentials)

# Predefined data
predefined_spectrograms = {
    'Muazzin Imran': cv2.imread("qari imran.jpg"),
    'Muazzin Mishary': cv2.imread("mufti meshiry.jpg"),
    'Muazzin Mufti Menk': cv2.imread("mufti menk.jpg"),
}

expert_audio_paths = {
    'Muazzin Imran': "Zohar macca.mp3",
    'Muazzin Mishary': "zohar by sheik mishary.mp3",
    'Muazzin Mufti Menk': "Zohar.mp3",
}

expert_transcriptions = {
    'Muazzin Imran': """الله أكبر الله أكبر الله أكبر الله أكبر أشهد أن لا إله إلا الله أشهد أن لا إله إلا الله أشهد أن محمدا رسول الله أشهد أن محمدا رسول الله حي على الصلاة حي على الصلاة حي على الفلاح حي على الفلاح الله أكبر الله أكبر لا إله إلا الله""",
    'Muazzin Mishary': """الله أكبر الله أكبر الله أكبر الله أكبر أشهد أن لا إله إلا الله أشهد أن لا إله إلا الله أشهد أن محمدا رسول الله أشهد أن محمدا رسول الله حي على الصلاة حي على الصلاة حي على الفلاح حي على الفلاح الله أكبر الله أكبر لا إله إلا الله""",
    'Muazzin Mufti Menk': """الله أكبر الله أكبر الله أكبر الله أكبر أشهد أن لا إله إلا الله أشهد أن لا إله إلا الله أشهد أن محمدا رسول الله أشهد أن محمدا رسول الله حي على الصلاة حي على الصلاة حي على الفلاح حي على الفلاح الله أكبر الله أكبر لا إله إلا الله"""
}

# Audio Feature Extraction and Analysis Functions
def extract_audio_features(audio_path):
    """Extract relevant audio features for analysis"""
    y, sr = librosa.load(audio_path)
    
    # Extract features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Calculate statistics
    pitch_stats = {
        'mean': np.mean(pitches[magnitudes > np.max(magnitudes) * 0.1]),
        'std': np.std(pitches[magnitudes > np.max(magnitudes) * 0.1]),
        'range': np.ptp(pitches[magnitudes > np.max(magnitudes) * 0.1])
    }
    
    return {
        'tempo': float(tempo),
        'pitch_stats': pitch_stats,
        'dynamics': {
            'mean_rms': float(np.mean(rms)),
            'max_rms': float(np.max(rms)),
            'dynamic_range': float(np.ptp(rms))
        },
        'articulation': float(np.mean(zero_crossings)),
        'timbre': {
            'mfcc_means': [float(np.mean(mfcc)) for mfcc in mfccs],
            'mfcc_stds': [float(np.std(mfcc)) for mfcc in mfccs]
        }
    }

import openai
import tempfile
import os
from pathlib import Path
import base64
import streamlit as st

def generate_audio_feedback(text):
    """
    Generate audio feedback using OpenAI's TTS API.
    """
    # Initialize OpenAI with your API key
    openai.api_key = st.secrets["openai"] # Replace with your actual OpenAI API Key
    try:
        # Create temporary file path
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "feedback_audio.mp3")

        # Generate speech using OpenAI API
        response = openai.Audio.create(
            model="tts-1",
            voice="shimmer",
            input=text
        )

        # Save the audio file
        with open(audio_path, 'wb') as audio_file:
            audio_file.write(response.content)

        return audio_path

    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

def create_audio_element(audio_path):
    """
    Create an HTML audio element for the feedback
    """
    try:
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode()
            
            return f"""
                <div style="background-color: #f5f5f5; padding: 10px; border-radius: 10px; margin: 10px 0;">
                    <audio controls style="width: 100%;">
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                        Your browser does not support the audio element.
                    </audio>
                </div>
            """
    except Exception as e:
        st.error(f"Error creating audio element: {str(e)}")
        return None

def display_audio_feedback(diagnosis_result):
    """
    Display the analysis results with audio feedback
    """
    if 'detailed_report' in diagnosis_result:
        sections = {
            "Pronunciation Analysis": "Let's start with your pronunciation analysis.",
            "Voice Quality": "Now, let's look at your voice quality.",
            "Special Points": "Here are some special points about your recitation.",
            "Overall Assessment": "Finally, here's your overall assessment."
        }

        st.markdown("""
            <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h3 style="color: #234e70; text-align: center;">Audio Feedback</h3>
            </div>
        """, unsafe_allow_html=True)

        for section_title, intro_text in sections.items():
            st.markdown(f"### {section_title}")
            
            # Generate audio feedback for this section
            feedback_text = f"{intro_text}\n{diagnosis_result['detailed_report']}"
            audio_path = generate_audio_feedback(feedback_text)
            
            if audio_path:
                audio_element = create_audio_element(audio_path)
                if audio_element:
                    st.markdown(audio_element, unsafe_allow_html=True)
                
                # Clean up temporary file
                try:
                    os.remove(audio_path)
                except:
                    pass
                
def display_detailed_report_with_audio(diagnosis_result):
    st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <h2 style="color: #234e70; text-align: center;">Detailed Analysis Report</h2>
        </div>
    """, unsafe_allow_html=True)

    # Display regular analysis
    display_pronunciation_analysis(diagnosis_result)
    display_voice_quality_analysis(diagnosis_result)
    
    # Add audio feedback
    display_audio_feedback(diagnosis_result)

    # Add download button for PDF report
    if 'detailed_report' in diagnosis_result:
        try:
            pdf_buffer = create_tafsili_report_pdf(diagnosis_result['detailed_report'])
            st.download_button(
                label="📥 Download Complete Report (PDF)",
                data=pdf_buffer,
                file_name="azaan_analysis_report.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")

def display_voice_quality_analysis(diagnosis_result):
    st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
            <h3 style="color: #234e70; margin-bottom: 1rem;">Awaaz ki Kaifiyat (Voice Quality Analysis)</h3>
            <div style="border-left: 4px solid #1ca4a4; padding-left: 1rem; margin: 1rem 0;">
    """, unsafe_allow_html=True)
    
    if 'audio_analysis' in diagnosis_result:
        metrics = {
            'Taal/Laye (Tempo)': diagnosis_result['audio_analysis']['tempo_diff'],
            'Pitch Accuracy': diagnosis_result['audio_analysis']['pitch_diff']['mean'],
            'Voice Control': diagnosis_result['audio_analysis']['dynamics_diff']['mean_rms'],
            'Overall Similarity': diagnosis_result['audio_analysis']['timbre_similarity']
        }
        
        for metric, value in metrics.items():
            quality = "Excellent" if value < 0.3 else "Good" if value < 0.6 else "Needs Improvement"
            color = "#1ca4a4" if value < 0.3 else "#d4af37" if value < 0.6 else "#dc3545"
            
            st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <strong>{metric}</strong>: 
                    <span style="color: {color};">{quality}</span>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
def display_detailed_report(diagnosis_result):
    st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); margin-top: 2rem;">
            <h3 style="color: #234e70; margin-bottom: 1rem; text-align: center;">
                <span style="border-bottom: 3px solid #1ca4a4; padding-bottom: 5px;">
                    Tafsili Report (Detailed Analysis)
                </span>
            </h3>
    """, unsafe_allow_html=True)
    
    if 'detailed_report' in diagnosis_result:
        # Display report content
        sections = diagnosis_result['detailed_report'].split('\n\n')
        for section in sections:
            if section.strip():
                if any(header in section.lower() for header in ['talaffuz', 'pronunciation', 'awaaz', 'voice', 'observations', 'mushahidaat', 'pattern', 'khaas nukaat']):
                    header_html = """
                        <div style="margin: 1.5rem 0 1rem 0;">
                            <h4 style="color: #1ca4a4; font-size: 1.2em; margin-bottom: 0.5rem;">
                                {0}
                            </h4>
                            <div style="background: #1ca4a4; height: 2px; width: 50px; margin: 0.5rem 0;"></div>
                            <div style="border-left: 4px solid #1ca4a4; padding-left: 1rem; margin-top: 1rem;">
                                {1}
                            </div>
                        </div>
                    """
                    header_content = section.split('\n')[0]
                    remaining_content = section.split('\n', 1)[1].replace('\n', '<br>') if '\n' in section else ''
                    st.markdown(header_html.format(header_content, remaining_content), unsafe_allow_html=True)
                else:
                    content_html = """
                        <div style="margin: 1rem 0; padding-left: 1rem;">
                            {0}
                        </div>
                    """
                    st.markdown(content_html.format(section.replace('\n', '<br>')), unsafe_allow_html=True)
        
        # Add download button right after the report content
        try:
            pdf_buffer = create_tafsili_report_pdf(diagnosis_result['detailed_report'])
            st.download_button(
                label="📥 Download Tafsili Report (PDF)",
                data=pdf_buffer,
                file_name="tafsili_report.pdf",
                mime="application/pdf",
                key="download_report"
            )
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)
def compare_audio_features(user_features, expert_features):
    """Compare user's audio features with expert's features"""
    differences = {
        'tempo_diff': abs(user_features['tempo'] - expert_features['tempo']),
        'pitch_diff': {
            'mean': abs(user_features['pitch_stats']['mean'] - expert_features['pitch_stats']['mean']),
            'std': abs(user_features['pitch_stats']['std'] - expert_features['pitch_stats']['std'])
        },
        'dynamics_diff': {
            'mean_rms': abs(user_features['dynamics']['mean_rms'] - expert_features['dynamics']['mean_rms']),
            'dynamic_range': abs(user_features['dynamics']['dynamic_range'] - expert_features['dynamics']['dynamic_range'])
        },
        'articulation_diff': abs(user_features['articulation'] - expert_features['articulation']),
        'timbre_similarity': pearsonr(
            user_features['timbre']['mfcc_means'],
            expert_features['timbre']['mfcc_means']
        )[0]
    }
    return differences

def analyze_pronunciation(transcription, expert_transcription):
    """Analyze pronunciation differences using text comparison"""
    words_user = transcription.split()
    words_expert = expert_transcription.split()
    
    differences = []
    for i, (word_user, word_expert) in enumerate(zip(words_user, words_expert)):
        if word_user != word_expert:
            differences.append({
                'word_number': i + 1,
                'user_word': word_user,
                'correct_word': word_expert
            })
    
    return differences

def generate_diagnosis_report(audio_comparison, pronunciation_analysis, closest_muezzin):
    """Generate an easy-to-understand diagnosis report in Roman Urdu"""
    
    # Define common issues and their simple explanations
    pronunciation_guide = {
        'الله': {
            'key_aspects': ['Allah lafz ka talaffuz', 'Laam ki gadrahat', 'Ha ki awaaz'],
            'explanation': 'Allah lafz mein Laam ko mota aur Ha ko halki awaaz'
        },
        'أكبر': {
            'key_aspects': ['Akbar mein Kaaf ki awaaz', 'Ber ki awaaz', 'Ra ki awaaz'],
            'explanation': 'Akbar mein Kaaf ko saaf aur Ra ko halka'
        },
        'أشهد': {
            'key_aspects': ['Ash ki awaaz', 'Ha ki awaaz', 'Dal ki awaaz'],
            'explanation': 'Ashhadu mein Ha ko halq se ada karna'
        },
        'حي': {
            'key_aspects': ['Ha ki awaaz', 'Ya ki awaaz'],
            'explanation': 'Hayya mein Ha ko halq se ada karna'
        },
        'على': {
            'key_aspects': ['Ain ki awaaz', 'Laam ki awaaz'],
            'explanation': 'Ala mein Ain ko gehri awaaz'
        }
    }

    prompt = f"""
    As an Azaan expert, provide a simple and clear diagnosis in Roman Urdu that any beginner can understand. Focus on explaining what was observed in the recitation without using technical terms.

    Audio Measurements:
    - Taal/Laye ka farq: {audio_comparison['tempo_diff']:.2f} 
    - Awaaz ki bulandee ka farq: {audio_comparison['pitch_diff']['mean']:.2f}
    - Awaaz ki yaksaniyat: {audio_comparison['dynamics_diff']['mean_rms']:.2f}
    - Talaffuz ki safai: {audio_comparison['articulation_diff']:.2f}
    - Awaaz ki mushababat: {audio_comparison['timbre_similarity']:.2f}

    Talaffuz ka Jaiza:
    {json.dumps(pronunciation_analysis, indent=2, ensure_ascii=False)}

    Please provide a detailed but simple analysis in the following format:

    1. Talaffuz (Pronunciation):
       - Halaq se nikalne wali awaazein (jaise Ha, Ain, Hamza)
       - Zuban se nikalne wali awaazein (jaise Ra, Laam)
       - Honthon se nikalne wali awaazein (jaise Meem, Ba)
       - Ghunna wali awaazein (Noon, Meem ki ghunna)

    2. Awaaz ki Kaifiyat (Voice Quality):
       - Awaaz ki bulandee
       - Awaaz ka utaar chadhao
       - Saansein lene ka tariqa
       - Awaaz ki safai

    3. Khaas Nukaat (Special Points):
       - Allah lafz ka talaffuz
       - Akbar mein Kaaf ki awaaz
       - Hayya mein Ha ki awaaz
       - Assalah aur Alfalah ke alfaaz

    4. Mushahidaat (Observations):
       - Khaas taqat ke pehlu
       - Behtar karne wale pehlu
       - Azaan ki yaksaniyat
       - Awaaz ki khususiyaat

    The response should be in simple Roman Urdu, easy for a common person to understand, and focus on clear observations without using technical Arabic terms.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "user", 
                "content": prompt
            }],
            temperature=0.7,
            max_tokens=1500,
            top_p=1.0
        )
        
        # Get initial analysis
        basic_analysis = response.choices[0].message["content"].strip()
        
        # Add simplified pattern analysis
        pattern_prompt = f"""
        Based on the above analysis, provide a simple explanation in Roman Urdu about:

        1. Azaan ki Tarteeб (Overall Flow):
           - Har jumle ka andaaz
           - Saans lene ke mawaqay
           - Jumlon ka rabt
           - Awaaz ka utaar chadhao

        2. Mukhtlif Hisson ka Jaiza (Different Parts):
           - Shuru ke jumlon ki kaifiyat
           - Darmiyan ke jumlon ki kaifiyat
           - Aakhri jumlon ki kaifiyat

        Use very simple language that a common person can understand easily.
        """
        
        pattern_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "assistant", "content": basic_analysis},
                {"role": "user", "content": pattern_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Combine into user-friendly report
        complete_diagnosis = f"""
Azan ke Talaffuz ka Jaiza
=========================

Bunyadi Jaiza:
-------------
{basic_analysis}

Tafsili Mushahidaat:
------------------
{pattern_response.choices[0].message["content"]}

Tehniki Paimaaish:
----------------
- Taal/Laye: {'Behtar' if audio_comparison['tempo_diff'] < 10 else 'Mazeed Behtar Ho Sakti Hai'}
- Awaaz ki Bulandee: {'Munasib' if audio_comparison['pitch_diff']['mean'] < 50 else 'Thodi Ziada'}
- Awaaz ki Yaksaniyat: {'Achhi' if audio_comparison['dynamics_diff']['mean_rms'] < 0.5 else 'Mazeed Behtar Ho Sakti Hai'}
- Talaffuz ki Safai: {'Waazeh' if audio_comparison['articulation_diff'] < 0.3 else 'Mazeed Waazeh Ho Sakti Hai'}
"""
        return complete_diagnosis
    
    except Exception as e:
        return f"Diagnosis report banane mein masla pesh aaya: {str(e)}"
        
def diagnose_azaan(user_audio_path, expert_audio_path, user_transcription, expert_transcription, closest_muezzin):
    """Main function to generate complete Azaan diagnosis"""
    
    # Extract audio features
    user_features = extract_audio_features(user_audio_path)
    expert_features = extract_audio_features(expert_audio_path)
    
    # Compare audio features
    audio_comparison = compare_audio_features(user_features, expert_features)
    
    # Analyze pronunciation
    pronunciation_analysis = analyze_pronunciation(user_transcription, expert_transcription)
    
    # Generate detailed report
    diagnosis_report = generate_diagnosis_report(audio_comparison, pronunciation_analysis, closest_muezzin)
    
    return {
        'audio_analysis': audio_comparison,
        'pronunciation_analysis': pronunciation_analysis,
        'detailed_report': diagnosis_report
    }

# Original Functions
def upload_audio_to_gcs(file, bucket_name="azaan_bucket"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_audio_file.write(file.getbuffer())
        temp_audio_file.flush()
        
        blob_name = f"demo/testing audios/{temp_audio_file.name.split('/')[-1]}"
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(temp_audio_file.name)
        
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        return gcs_uri

def download_audio_from_gcs(gcs_uri):
    bucket_name = gcs_uri.split("/")[2]
    blob_name = "/".join(gcs_uri.split("/")[3:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        blob.download_to_filename(temp_audio_file.name)
        return temp_audio_file.name

def transcribe_audio_gcs(gcs_uri):
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=48000,
        language_code="ar"
    )
    operation = speech_client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=180)
    full_transcript = " ".join([result.alternatives[0].transcript for result in response.results])
    return full_transcript.strip()

def normalize_text(text):
    text = re.sub(r'[ًٌٍَُِ]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه")
    return text

def validate_with_openai(transcription):
    correct_azaan = """
    الله أكبر الله أكبر
    الله أكبر الله أكبر
    أشهد أن لا إله إلا الله
    أشهد أن لا إله إلا الله
    أشهد أن محمدًا رسول الله
    أشهد أن محمدًا رسول الله
    حي على الصلاة
    حي على الصلاة
    حي على الفلاح
    حي على الفلاح
    الله أكبر الله أكبر
    لا إله إلا الله
    """

    prompt = f"""
    You are an expert in validating the Azaan (the call to prayer). Below is the correct structure of the Azaan. 
    Compare the transcription provided with this structure to determine if it contains all essential phrases in the correct order.

    Validation Guidelines:
    - Validate the Azaan as "VALIDATED" if it contains all essential phrases in the correct sequence, even if there are minor spelling, diacritic, or punctuation differences.
    - Specifically, ignore small differences such as:
        - Missing or extra diacritics (e.g., "ا" vs. "أ" or "حي على الصلاه" vs. "حي على الصلاة").
        - Minor spelling variations, such as:
            - "لا اله الا الله" vs. "لا إله إلا الله".
            - "حي على الصلاه" vs. "حي على الصلاة".
            - "حي على الفلاح" vs. "حي على الفلاح".
            - "أشهد" vs "شهاده"
        - Punctuation or slight variations in commonly understood words and phrases.
    - Invalidate the Azaan as "INVALIDATED" only if:
        - Essential phrases are missing.
        - Extra, unrelated phrases that are not part of the Azaan are added.
        - Major incorrect words or substitutions that change the meaning of an essential phrase are present.

    Correct Azaan Structure:
    {correct_azaan}

    Transcribed Azaan:
    "{transcription}"

    Conclude with "Validation Status: VALIDATED" if the Azaan matches the correct structure, or "Validation Status: INVALIDATED" if it does not, and list any specific issues if found. Only list issues if they involve missing phrases, extra phrases, or significant meaning changes.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250,
            top_p=1.0,
            stop=None
        )
        validation_result = response.choices[0].message["content"].strip()
        st.write("Response:", validation_result)
        return validation_result
    except Exception as e:
        st.error(f"Error with OpenAI API request: {e}")
        return None

def generate_mel_spectrogram(audio_file):
    y, sr = librosa.load(audio_file, sr=22050)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db, sr

def plot_spectrogram(mel_spectrogram, sr):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mel_spectrogram, sr=sr, x_axis='time', y_axis='mel', cmap='magma', ax=ax)
    ax.set(title="Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    plt.close(fig)
    return buf

def calculate_similarity(image1, image2):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGBA2GRAY) if image1.shape[2] == 4 else cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGBA2GRAY) if image2.shape[2] == 4 else cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    if image1_gray.shape != image2_gray.shape:
        image2_gray = cv2.resize(image2_gray, (image1_gray.shape[1], image1_gray.shape[0]))
    return np.mean((image1_gray - image2_gray) ** 2)

import streamlit as st
from streamlit.components.v1 import html
import base64
from pathlib import Path

def set_page_config():
    st.set_page_config(
        page_title="Azaan Analysis System",
        page_icon="🕌",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

def load_css():
    st.markdown("""
        <style>
        /* Main container styling */
        .stApp {
            background: linear-gradient(135deg, #f4e4bc, #fff);
        }
        
        /* Header styling */
        .main-header {
            text-align: center;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            margin-bottom: 2rem;
        }
        
        /* Custom upload button */
        .stButton>button {
            background-color: #1ca4a4;
            color: white;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #234e70;
            transform: translateY(-2px);
        }
        
        /* Progress bars */
        .stProgress > div > div {
            background-color: #1ca4a4;
        }
        
        /* Cards styling */
        .css-1r6slb0 {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        /* Metrics styling */
        .css-1r6slb0.e16fv1kl3 {
            background: white;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
import arabic_reshaper
from bidi.algorithm import get_display
import os
import io

def setup_arabic_font():
    # Register Arabic font - ensure this font file exists in your project
    font_path = r"C:\Users\USER\Downloads\Amiri-Regular.ttf"  # Update with actual path
    pdfmetrics.registerFont(TTFont('Arabic', font_path))
    
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import datetime
import uuid
def create_tafsili_report_pdf(detailed_report):
    buffer = io.BytesIO()
    
    # Register Amiri font for Arabic text
    amiri_font_path = r"C:\Users\USER\Downloads\Amiri-Regular.ttf"
    pdfmetrics.registerFont(TTFont('Amiri', amiri_font_path))
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='ArabicStyle',
        fontName='Amiri',
        fontSize=12,
        alignment=2,
        leading=16
    ))
    
    story = []
    sections = detailed_report.split('\n')
    
    for section in sections:
        if section.strip():
            # Handle Arabic text
            if any('\u0600' <= c <= '\u06FF' for c in section):
                reshaped_text = arabic_reshaper.reshape(section)
                bidi_text = get_display(reshaped_text)
                p = Paragraph(bidi_text, styles['ArabicStyle'])
            else:
                # Non-Arabic text uses normal style
                p = Paragraph(section, styles['Normal'])
            story.append(p)
            story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_temporary_url(pdf_buffer):
    # Save PDF to temporary file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "report.pdf")
    
    with open(temp_path, 'wb') as f:
        f.write(pdf_buffer.getvalue())
    
    # Upload to Google Cloud Storage with expiration
    bucket = storage_client.bucket("azaan_bucket")
    blob = bucket.blob(f"reports/{uuid.uuid4()}.pdf")
    blob.upload_from_filename(temp_path)
    
    # Generate signed URL that expires in 1 hour
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(hours=1),
        method="GET"
    )
    
    return url

def add_download_button(detailed_report):
    pdf_buffer = create_tafsili_report_pdf(detailed_report)
    temp_url = generate_temporary_url(pdf_buffer)
    
    st.markdown(f'''
        <a href="{temp_url}" target="_blank" class="view-pdf">
            <button style="background-color: #1ca4a4; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
                View Report
            </button>
        </a>
    ''', unsafe_allow_html=True)
    
    st.download_button(
        label="📥 Download Report",
        data=pdf_buffer,
        file_name="tafsili_report.pdf",
        mime="application/pdf"
    )
def main():
   set_page_config()
   load_css()
   
   st.markdown("""
       <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(244,228,188,0.9)); 
                   border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
           <h1 style="color: #234e70; font-size: 2.5em; margin-bottom: 0.5rem;">🕌 Azaan Analysis System</h1>
           <p style="color: #4a4238; font-size: 1.2em; font-style: italic;">
               "Indeed, the most beautiful among you are those who have the most beautiful voices in reciting the Azaan"
           </p>
           <div style="background: #1ca4a4; height: 3px; width: 150px; margin: 1rem auto;"></div>
       </div>
   """, unsafe_allow_html=True)

   st.markdown("""
       <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
                   text-align: center; margin-bottom: 2rem;">
           <h2 style="color: #1ca4a4; margin-bottom: 1rem;">Upload Your Azaan Recording</h2>
           <p style="color: #4a4238;">Please upload your Azaan recording in MP3 format</p>
       </div>
   """, unsafe_allow_html=True)

   uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3"])
   
   if uploaded_file:
       progress_placeholder = st.empty()
       progress_bar = st.progress(0)
       status_text = st.empty()
       
       try:
           with st.spinner("📤 Uploading audio..."):
               gcs_uri = upload_audio_to_gcs(uploaded_file)
               progress_bar.progress(25)
               status_text.text("✅ Audio uploaded successfully")
           
           with st.spinner("🔍 Transcribing and validating..."):
               transcription = transcribe_audio_gcs(gcs_uri)
               validation_result = validate_with_openai(transcription)
               progress_bar.progress(50)
               status_text.text("✅ Transcription and validation complete")
           
           if "Validation Status: VALIDATED" in validation_result:
               with st.spinner("⚡ Analyzing audio patterns..."):
                   downloaded_audio_path = download_audio_from_gcs(gcs_uri)
                   progress_bar.progress(75)
                   
                   user_spectrogram, user_sr = generate_mel_spectrogram(downloaded_audio_path)
                   user_spectrogram_img = plot_spectrogram(user_spectrogram, user_sr)
                   
                   if user_spectrogram_img is not None:
                       progress_bar.progress(85)
                       status_text.text("📊 Generating detailed analysis...")
                       
                       user_image = np.array(Image.open(user_spectrogram_img))
                       similarities = {}
                       
                       for muezzin, predefined_spectrogram in predefined_spectrograms.items():
                           if predefined_spectrogram is not None:
                               similarity_score = calculate_similarity(user_image, predefined_spectrogram)
                               similarities[muezzin] = similarity_score
                       
                       if similarities:
                           closest_muezzin = min(similarities, key=similarities.get)
                           
                           diagnosis_result = diagnose_azaan(
                               downloaded_audio_path,
                               expert_audio_paths[closest_muezzin],
                               transcription,
                               expert_transcriptions[closest_muezzin],
                               closest_muezzin
                           )
                           
                           progress_bar.progress(100)
                           status_text.text("✨ Analysis complete!")
                           
                           st.markdown("""
                               <div style="text-align: center; margin: 2rem 0;">
                                   <h2 style="color: #1ca4a4;">Analysis Results</h2>
                                   <div style="background: #1ca4a4; height: 3px; width: 100px; margin: 1rem auto;"></div>
                               </div>
                           """, unsafe_allow_html=True)
                           
                           st.markdown(f"""
                               <div style="background: white; padding: 1.5rem; border-radius: 15px; 
                                         box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center; margin-bottom: 2rem;">
                                   <h3 style="color: #234e70;">Matching Analysis</h3>
                                   <p style="color: #1ca4a4; font-size: 1.2em; margin: 1rem 0;">
                                       MashaAllah! Your recitation style closely matches {closest_muezzin}'s pattern
                                   </p>
                               </div>
                           """, unsafe_allow_html=True)
                           
                           display_voice_quality_analysis(diagnosis_result)
                           
                           if 'detailed_report' in diagnosis_result:
                               st.markdown("""
                                   <div style="margin: 2rem 0;">
                                       <h3 style="color: #234e70;">Detailed Report</h3>
                                   </div>
                               """, unsafe_allow_html=True)
                               
                               # Generate PDF and temporary URL
                               pdf_buffer = create_tafsili_report_pdf(diagnosis_result['detailed_report'])
                               temp_url = generate_temporary_url(pdf_buffer)
                               
                               # Add view button that opens in new tab
                               st.markdown(f'''
                                   <a href="{temp_url}" target="_blank" class="view-pdf">
                                       <button style="background-color: #1ca4a4; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin-right: 10px;">
                                           View Report
                                       </button>
                                   </a>
                               ''', unsafe_allow_html=True)
                               
                               # Add download button
                               st.download_button(
                                   label="📥 Download Report",
                                   data=pdf_buffer,
                                   file_name="tafsili_report.pdf",
                                   mime="application/pdf"
                               )
                           
                           st.markdown("""
                               <div style="text-align: center; margin-top: 3rem; padding: 1rem; color: #4a4238;">
                                   <p>May Allah accept your efforts in perfecting the call to prayer.</p>
                                   <div style="background: #1ca4a4; height: 2px; width: 100px; margin: 1rem auto;"></div>
                               </div>
                           """, unsafe_allow_html=True)
                       
                       else:
                           st.error("Unable to perform pattern matching. Please try again.")
                   else:
                       st.error("Error generating audio visualization.")
           else:
               st.error("""
                   Validation Failed! Please ensure your recording follows the correct Azaan structure.
                   Check if all essential phrases are present and in the correct order.
               """)
       
       except Exception as e:
           st.error(f"An error occurred during analysis: {str(e)}")
           progress_bar.empty()
           status_text.empty()

if __name__ == "__main__":
   main()
