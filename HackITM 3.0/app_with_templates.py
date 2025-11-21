# AI Resume Generator with Templates - Experimental Version
import streamlit as st
import pdfplumber
import docx
import re
import random
import io
from datetime import datetime
import base64

# Fix for reportlab compatibility
import hashlib
_original_md5 = hashlib.md5
def _patched_md5(data=None, **kwargs):
    if 'usedforsecurity' in kwargs:
        try:
            if data is None:
                return _original_md5(**kwargs)
            return _original_md5(data, **kwargs)
        except (TypeError, ValueError):
            kwargs = {k: v for k, v in kwargs.items() if k != 'usedforsecurity'}
            if data is None:
                return _original_md5(**kwargs) if kwargs else _original_md5()
            return _original_md5(data, **kwargs) if kwargs else _original_md5(data)
    if data is None:
        return _original_md5(**kwargs) if kwargs else _original_md5()
    return _original_md5(data, **kwargs) if kwargs else _original_md5(data)
hashlib.md5 = _patched_md5

_original_new = hashlib.new
def _patched_new(name, data=b'', **kwargs):
    if 'usedforsecurity' in kwargs:
        try:
            return _original_new(name, data, **kwargs)
        except (TypeError, ValueError):
            kwargs = {k: v for k, v in kwargs.items() if k != 'usedforsecurity'}
            return _original_new(name, data, **kwargs) if kwargs else _original_new(name, data)
    return _original_new(name, data, **kwargs) if kwargs else _original_new(name, data)
hashlib.new = _patched_new

# Import reportlab
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas

class ResumeTemplate:
    """Base class for resume templates"""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    def parse_resume_text(self, resume_text):
        """Parse resume text into structured sections"""
        sections = {
            'header': {'name': '', 'email': '', 'phone': '', 'location': '', 'linkedin': '', 'portfolio': ''},
            'summary': '',
            'experience': [],
            'education': [],
            'skills': [],
            'projects': [],
            'certifications': [],
            'awards': []
        }
        
        lines = resume_text.split('\n')
        current_section = None
        current_item = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            line_upper = line.upper()
            if 'PROFESSIONAL SUMMARY' in line_upper or 'SUMMARY' in line_upper:
                current_section = 'summary'
                continue
            elif 'EXPERIENCE' in line_upper or 'WORK EXPERIENCE' in line_upper or 'EMPLOYMENT' in line_upper:
                current_section = 'experience'
                continue
            elif 'EDUCATION' in line_upper:
                current_section = 'education'
                continue
            elif 'SKILLS' in line_upper or 'TECHNICAL SKILLS' in line_upper:
                current_section = 'skills'
                continue
            elif 'PROJECTS' in line_upper:
                current_section = 'projects'
                continue
            elif 'CERTIFICATIONS' in line_upper or 'CERTIFICATES' in line_upper:
                current_section = 'certifications'
                continue
            elif 'AWARDS' in line_upper or 'ACHIEVEMENTS' in line_upper:
                current_section = 'awards'
                continue
            
            # Parse header information (first few lines before sections)
            if current_section is None:
                # Try to extract name, email, phone, etc.
                if '@' in line and '.' in line:
                    sections['header']['email'] = line
                elif re.match(r'[\d\s\-\(\)]+', line) and len(line.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')) >= 10:
                    sections['header']['phone'] = line
                elif 'linkedin.com' in line.lower():
                    sections['header']['linkedin'] = line
                elif 'http' in line.lower() or 'www.' in line.lower():
                    sections['header']['portfolio'] = line
                elif not sections['header']['name'] and len(line) < 50 and not any(c in line for c in ['|', ',']):
                    sections['header']['name'] = line
                elif '|' in line:
                    # Parse header line with separators
                    parts = [p.strip() for p in line.split('|')]
                    for part in parts:
                        if '@' in part:
                            sections['header']['email'] = part
                        elif re.match(r'[\d\s\-\(\)]+', part):
                            sections['header']['phone'] = part
                        elif 'linkedin' in part.lower():
                            sections['header']['linkedin'] = part
                        else:
                            sections['header']['location'] = part
            
            # Parse section content
            elif current_section == 'summary':
                if sections['summary']:
                    sections['summary'] += ' ' + line
                else:
                    sections['summary'] = line
            elif current_section == 'experience':
                if line.startswith(('•', '-', '*')):
                    if current_item:
                        sections['experience'].append(current_item)
                    current_item = {'description': line.lstrip('•-*').strip(), 'title': '', 'company': '', 'dates': ''}
                elif '|' in line or ' - ' in line:
                    # Job title and company
                    parts = re.split(r'\s*-\s*|\s*\|\s*', line)
                    if len(parts) >= 2:
                        current_item['title'] = parts[0].strip()
                        current_item['company'] = parts[1].strip()
                        if len(parts) >= 3:
                            current_item['dates'] = parts[2].strip()
                elif current_item:
                    current_item['description'] += ' ' + line
            elif current_section == 'education':
                sections['education'].append(line)
            elif current_section == 'skills':
                # Skills can be comma-separated or line-separated
                if ',' in line:
                    sections['skills'].extend([s.strip() for s in line.split(',')])
                else:
                    sections['skills'].append(line.lstrip('•-*').strip())
            elif current_section == 'projects':
                sections['projects'].append(line)
            elif current_section == 'certifications':
                sections['certifications'].append(line)
            elif current_section == 'awards':
                sections['awards'].append(line)
        
        # Add last experience item
        if current_section == 'experience' and current_item:
            sections['experience'].append(current_item)
        
        return sections
    
    def generate_pdf(self, resume_text):
        """Generate PDF - to be implemented by subclasses"""
        raise NotImplementedError

class ModernTemplate(ResumeTemplate):
    """Modern, clean template with colored header"""
    
    def __init__(self):
        super().__init__("Modern", "Clean design with colored header section")
    
    def generate_pdf(self, resume_text):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              rightMargin=0.75*inch, leftMargin=0.75*inch,
                              topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        sections = self.parse_resume_text(resume_text)
        story = []
        styles = getSampleStyleSheet()
        
        # Header style with background color
        header_style = ParagraphStyle(
            'Header',
            parent=styles['Normal'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=6,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER
        )
        
        contact_style = ParagraphStyle(
            'Contact',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12,
            alignment=TA_CENTER
        )
        
        section_style = ParagraphStyle(
            'Section',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#3498db'),
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold',
            borderWidth=0,
            borderPadding=0,
            borderColor=colors.HexColor('#3498db'),
            leftIndent=0
        )
        
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.black,
            spaceAfter=4,
            leading=12
        )
        
        bullet_style = ParagraphStyle(
            'Bullet',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.black,
            spaceAfter=3,
            leftIndent=15,
            bulletIndent=5
        )
        
        # Header
        if sections['header']['name']:
            story.append(Paragraph(sections['header']['name'].upper(), header_style))
        
        # Contact info
        contact_parts = []
        if sections['header']['email']:
            contact_parts.append(sections['header']['email'])
        if sections['header']['phone']:
            contact_parts.append(sections['header']['phone'])
        if sections['header']['location']:
            contact_parts.append(sections['header']['location'])
        if sections['header']['linkedin']:
            contact_parts.append(sections['header']['linkedin'])
        
        if contact_parts:
            contact_text = ' | '.join(contact_parts)
            story.append(Paragraph(contact_text, contact_style))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Professional Summary
        if sections['summary']:
            story.append(Paragraph("PROFESSIONAL SUMMARY", section_style))
            story.append(Paragraph(sections['summary'], normal_style))
            story.append(Spacer(1, 0.15*inch))
        
        # Experience
        if sections['experience']:
            story.append(Paragraph("PROFESSIONAL EXPERIENCE", section_style))
            for exp in sections['experience'][:5]:  # Limit to 5 experiences
                if exp.get('title'):
                    title_text = f"<b>{exp['title']}</b>"
                    if exp.get('company'):
                        title_text += f" | {exp['company']}"
                    if exp.get('dates'):
                        title_text += f" | {exp['dates']}"
                    story.append(Paragraph(title_text, normal_style))
                if exp.get('description'):
                    desc = exp['description'].replace('•', '•').replace('-', '•')
                    story.append(Paragraph(f"• {desc}", bullet_style))
                story.append(Spacer(1, 0.1*inch))
            story.append(Spacer(1, 0.1*inch))
        
        # Skills
        if sections['skills']:
            story.append(Paragraph("SKILLS", section_style))
            skills_text = ', '.join(sections['skills'][:20])  # Limit skills
            story.append(Paragraph(skills_text, normal_style))
            story.append(Spacer(1, 0.15*inch))
        
        # Education
        if sections['education']:
            story.append(Paragraph("EDUCATION", section_style))
            for edu in sections['education'][:3]:  # Limit to 3
                story.append(Paragraph(edu, normal_style))
            story.append(Spacer(1, 0.15*inch))
        
        # Projects
        if sections['projects']:
            story.append(Paragraph("PROJECTS", section_style))
            for proj in sections['projects'][:3]:  # Limit to 3
                story.append(Paragraph(f"• {proj.lstrip('•-*')}", bullet_style))
        
        doc.build(story)
        buffer.seek(0)
        return buffer

class ClassicTemplate(ResumeTemplate):
    """Classic, traditional template"""
    
    def __init__(self):
        super().__init__("Classic", "Traditional format with clear sections")
    
    def generate_pdf(self, resume_text):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              rightMargin=1*inch, leftMargin=1*inch,
                              topMargin=1*inch, bottomMargin=1*inch)
        
        sections = self.parse_resume_text(resume_text)
        story = []
        styles = getSampleStyleSheet()
        
        # Styles
        name_style = ParagraphStyle(
            'Name',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.black,
            spaceAfter=6,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER
        )
        
        contact_style = ParagraphStyle(
            'Contact',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.black,
            spaceAfter=15,
            alignment=TA_CENTER
        )
        
        section_style = ParagraphStyle(
            'Section',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.black,
            spaceAfter=6,
            spaceBefore=10,
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderPadding=2,
            borderColor=colors.black
        )
        
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.black,
            spaceAfter=4,
            leading=12
        )
        
        # Header
        if sections['header']['name']:
            story.append(Paragraph(sections['header']['name'], name_style))
        
        contact_parts = []
        if sections['header']['email']:
            contact_parts.append(sections['header']['email'])
        if sections['header']['phone']:
            contact_parts.append(sections['header']['phone'])
        if sections['header']['location']:
            contact_parts.append(sections['header']['location'])
        
        if contact_parts:
            story.append(Paragraph(' | '.join(contact_parts), contact_style))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Sections
        if sections['summary']:
            story.append(Paragraph("PROFESSIONAL SUMMARY", section_style))
            story.append(Paragraph(sections['summary'], normal_style))
            story.append(Spacer(1, 0.15*inch))
        
        if sections['experience']:
            story.append(Paragraph("EXPERIENCE", section_style))
            for exp in sections['experience'][:5]:
                if exp.get('title'):
                    title_text = f"<b>{exp['title']}</b>"
                    if exp.get('company'):
                        title_text += f" - {exp['company']}"
                    if exp.get('dates'):
                        title_text += f" ({exp['dates']})"
                    story.append(Paragraph(title_text, normal_style))
                if exp.get('description'):
                    story.append(Paragraph(f"• {exp['description']}", normal_style))
                story.append(Spacer(1, 0.08*inch))
        
        if sections['skills']:
            story.append(Paragraph("SKILLS", section_style))
            skills_text = ', '.join(sections['skills'][:20])
            story.append(Paragraph(skills_text, normal_style))
            story.append(Spacer(1, 0.15*inch))
        
        if sections['education']:
            story.append(Paragraph("EDUCATION", section_style))
            for edu in sections['education'][:3]:
                story.append(Paragraph(edu, normal_style))
        
        doc.build(story)
        buffer.seek(0)
        return buffer

class CreativeTemplate(ResumeTemplate):
    """Creative template with side bar"""
    
    def __init__(self):
        super().__init__("Creative", "Modern design with sidebar layout")
    
    def generate_pdf(self, resume_text):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              rightMargin=0.5*inch, leftMargin=0.5*inch,
                              topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        sections = self.parse_resume_text(resume_text)
        story = []
        styles = getSampleStyleSheet()
        
        # Create a table for sidebar layout
        data = []
        
        # Left column (sidebar) - 2 inches
        left_col = []
        if sections['header']['name']:
            name_style = ParagraphStyle(
                'Name',
                fontSize=20,
                textColor=colors.white,
                fontName='Helvetica-Bold',
                alignment=TA_CENTER
            )
            left_col.append(Paragraph(sections['header']['name'].upper(), name_style))
            left_col.append(Spacer(1, 0.2*inch))
        
        # Contact in sidebar
        contact_style = ParagraphStyle(
            'Contact',
            fontSize=9,
            textColor=colors.white,
            spaceAfter=4
        )
        
        if sections['header']['email']:
            left_col.append(Paragraph(sections['header']['email'], contact_style))
        if sections['header']['phone']:
            left_col.append(Paragraph(sections['header']['phone'], contact_style))
        if sections['header']['location']:
            left_col.append(Paragraph(sections['header']['location'], contact_style))
        
        left_col.append(Spacer(1, 0.3*inch))
        
        # Skills in sidebar
        if sections['skills']:
            skill_header = ParagraphStyle(
                'SkillHeader',
                fontSize=12,
                textColor=colors.white,
                fontName='Helvetica-Bold',
                spaceAfter=6
            )
            left_col.append(Paragraph("SKILLS", skill_header))
            for skill in sections['skills'][:15]:
                left_col.append(Paragraph(f"• {skill}", contact_style))
        
        # Right column (main content) - 4.5 inches
        right_col = []
        
        section_style = ParagraphStyle(
            'Section',
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            fontName='Helvetica-Bold',
            spaceAfter=6,
            spaceBefore=12
        )
        
        normal_style = ParagraphStyle(
            'Normal',
            fontSize=10,
            textColor=colors.black,
            spaceAfter=4,
            leading=12
        )
        
        if sections['summary']:
            right_col.append(Paragraph("PROFESSIONAL SUMMARY", section_style))
            right_col.append(Paragraph(sections['summary'], normal_style))
            right_col.append(Spacer(1, 0.15*inch))
        
        if sections['experience']:
            right_col.append(Paragraph("EXPERIENCE", section_style))
            for exp in sections['experience'][:4]:
                if exp.get('title'):
                    title_text = f"<b>{exp['title']}</b>"
                    if exp.get('company'):
                        title_text += f" | {exp['company']}"
                    right_col.append(Paragraph(title_text, normal_style))
                if exp.get('description'):
                    right_col.append(Paragraph(f"• {exp['description']}", normal_style))
                right_col.append(Spacer(1, 0.1*inch))
        
        if sections['education']:
            right_col.append(Paragraph("EDUCATION", section_style))
            for edu in sections['education'][:2]:
                right_col.append(Paragraph(edu, normal_style))
        
        # Create table with sidebar
        table_data = [[left_col, right_col]]
        table = Table(table_data, colWidths=[2*inch, 4.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#34495e')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        story.append(table)
        
        doc.build(story)
        buffer.seek(0)
        return buffer

def generate_pdf_with_template(resume_text, template_name="Modern"):
    """Generate PDF using selected template"""
    templates = {
        "Modern": ModernTemplate(),
        "Classic": ClassicTemplate(),
        "Creative": CreativeTemplate()
    }
    
    template = templates.get(template_name, templates["Modern"])
    return template.generate_pdf(resume_text)

def main():
    st.set_page_config(
        page_title="AI Resume Generator with Templates",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .template-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s;
    }
    .template-card:hover {
        border-color: #1f77b4;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🎨 AI Resume Generator with Templates</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Options")
        template_option = st.selectbox(
            "Choose Template",
            ["Modern", "Classic", "Creative"],
            help="Select a resume template style"
        )
        
        st.header("⚙️ Settings")
        job_title = st.text_input("Target Job Title", "Software Engineer")
        job_description = st.text_area("Job Description", height=150)
    
    # Main content
    st.header("📝 Resume Content")
    
    resume_content = st.text_area(
        "Enter or paste your resume content here",
        height=400,
        help="Paste your resume text. The template will automatically format it."
    )
    
    if resume_content:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Preview")
            st.text_area("Resume Preview", resume_content, height=300, disabled=True)
        
        with col2:
            st.subheader("🎨 Template Preview")
            template_descriptions = {
                "Modern": "Clean design with colored header section",
                "Classic": "Traditional format with clear sections",
                "Creative": "Modern design with sidebar layout"
            }
            st.info(f"**Selected Template:** {template_option}")
            st.caption(template_descriptions.get(template_option, ""))
            
            if st.button("🔄 Generate PDF Preview", type="primary", use_container_width=True):
                with st.spinner("Generating PDF with template..."):
                    try:
                        pdf_buffer = generate_pdf_with_template(resume_content, template_option)
                        pdf_bytes = pdf_buffer.getvalue()
                        
                        # Display PDF
                        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")
        
        st.markdown("---")
        st.header("💾 Download")
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            if st.button("📄 Download PDF", type="primary", use_container_width=True):
                try:
                    pdf_buffer = generate_pdf_with_template(resume_content, template_option)
                    pdf_bytes = pdf_buffer.getvalue()
                    st.download_button(
                        label="⬇️ Download PDF File",
                        data=pdf_bytes,
                        file_name=f"resume_{template_option.lower()}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col_d2:
            # Template information
            st.info(f"""
            **Template Features:**
            - Automatic formatting
            - Content fitting
            - Professional layout
            - ATS-friendly structure
            """)

if __name__ == "__main__":
    main()

