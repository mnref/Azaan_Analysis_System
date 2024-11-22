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
import streamlit as st

def setup_arabic_font():
    # Register Arabic font - ensure this font file exists in your project
    font_path = "path/to/your/arabic-font.ttf"  # Update with actual path
    pdfmetrics.registerFont(TTFont('Arabic', font_path))
    
def create_tafsili_report_pdf(detailed_report):
    # Setup buffer for PDF
    buffer = io.BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    # Setup styles
    styles = getSampleStyleSheet()
    arabic_style = ParagraphStyle(
        'ArabicStyle',
        parent=styles['Normal'],
        fontName='Arabic',
        fontSize=12,
        rightIndent=0,
        leftIndent=0,
        alignment=2  # Right alignment for Arabic
    )
    
    # Process content
    story = []
    
    # Split report into sections
    sections = detailed_report.split('\n\n')
    
    for section in sections:
        if section.strip():
            # Process Arabic text
            lines = section.split('\n')
            for line in lines:
                if any('\u0600' <= c <= '\u06FF' for c in line):
                    # Arabic text processing
                    reshaped_text = arabic_reshaper.reshape(line)
                    bidi_text = get_display(reshaped_text)
                    p = Paragraph(bidi_text, arabic_style)
                else:
                    # Non-Arabic text
                    p = Paragraph(line, styles['Normal'])
                story.append(p)
                story.append(Spacer(1, 12))
    
    # Build PDF
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