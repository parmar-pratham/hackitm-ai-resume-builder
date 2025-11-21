# AI Resume Generator with PDF Upload & Editing
import streamlit as st
import pdfplumber
try:
    import docx
except (ImportError, ModuleNotFoundError) as e:
    error_msg = """
    ============================================
    IMPORT ERROR: docx package conflict detected
    ============================================
    
    You have an old Python 2 'docx' package installed that conflicts with 'python-docx'.
    
    To fix this, run these commands in your terminal/command prompt:
    
        pip uninstall docx
        pip install python-docx
    
    Then restart the application.
    
    Original error: {}
    """.format(str(e))
    print(error_msg)
    raise ImportError(
        "Please uninstall the old 'docx' package and install 'python-docx'. "
        "Run: pip uninstall docx && pip install python-docx"
    ) from e
import re
import random
import io
from datetime import datetime
import base64

# Fix for reportlab compatibility with newer Python versions
# This fixes the 'usedforsecurity' error with openssl_md5()
import hashlib
import sys

# Patch hashlib.md5 to gracefully handle usedforsecurity parameter
_original_md5 = hashlib.md5
def _patched_md5(data=None, **kwargs):
    # Remove usedforsecurity if it's not supported by the underlying implementation
    if 'usedforsecurity' in kwargs:
        try:
            # Try with usedforsecurity first
            if data is None:
                return _original_md5(**kwargs)
            return _original_md5(data, **kwargs)
        except (TypeError, ValueError) as e:
            # If it fails, remove usedforsecurity and try again
            kwargs = {k: v for k, v in kwargs.items() if k != 'usedforsecurity'}
            if data is None:
                return _original_md5(**kwargs) if kwargs else _original_md5()
            return _original_md5(data, **kwargs) if kwargs else _original_md5(data)
    # No usedforsecurity, call normally
    if data is None:
        return _original_md5(**kwargs) if kwargs else _original_md5()
    return _original_md5(data, **kwargs) if kwargs else _original_md5(data)

hashlib.md5 = _patched_md5

# Patch hashlib.new as well
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

# Now import reportlab after the patch
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.units import inch
from reportlab.lib import colors

# Ollama AI Integration
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    st.warning("requests library not found. Install with: pip install requests")

class ResumeProcessor:
    def __init__(self):
        self.skill_keywords = self._load_skill_keywords()
        self.impact_phrases = [
            "resulting in improved efficiency and productivity",
            "leading to significant cost savings",
            "increasing performance by measurable margins", 
            "enhancing user experience and satisfaction",
            "streamlining processes and reducing overhead",
            "driving growth and innovation"
        ]
        self.action_verbs = [
            'developed', 'implemented', 'managed', 'led', 'created', 'designed',
            'optimized', 'analyzed', 'improved', 'increased', 'reduced', 'automated',
            'built', 'engineered', 'architected', 'coordinated', 'facilitated'
        ]
    
    def _load_skill_keywords(self):
        """Load common technical and soft skills"""
        technical_skills = [
            'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'node.js',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'aws',
            'docker', 'kubernetes', 'git', 'jenkins', 'ci/cd', 'rest api', 'graphql',
            'mongodb', 'postgresql', 'mysql', 'linux', 'bash', 'powershell',
            'data analysis', 'data visualization', 'pandas', 'numpy', 'tableau',
            'power bi', 'excel', 'r', 'scala', 'c++', 'c#', 'php', 'ruby', 'go'
        ]
        
        soft_skills = [
            'communication', 'leadership', 'teamwork', 'problem solving', 'critical thinking',
            'adaptability', 'time management', 'creativity', 'collaboration', 'analytical skills',
            'project management', 'agile', 'scrum', 'presentation', 'negotiation'
        ]
        
        return technical_skills + soft_skills
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
        text = ""
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            st.error(f"Error extracting PDF: {str(e)}")
        return text
    
    def extract_text_from_docx(self, docx_file):
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(docx_file)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
        except Exception as e:
            st.error(f"Error extracting DOCX: {str(e)}")
            return ""
    
    def extract_skills_from_jd(self, job_description):
        """Extract skills from job description"""
        if not job_description:
            return []
            
        jd_lower = job_description.lower()
        found_skills = []
        
        # Match skills from predefined list
        for skill in self.skill_keywords:
            if skill.lower() in jd_lower:
                found_skills.append(skill)
        
        # Extract skills using regex patterns
        skill_patterns = [
            r'\b(?:proficient in|experience with|knowledge of|skills in|expertise in)\s+([^.,]+)',
            r'\b(?:python|java|javascript|sql|html|css|react|node\.js|typescript)\b',
            r'\b(?:aws|azure|gcp|docker|kubernetes|terraform)\b',
            r'\b(?:machine learning|deep learning|ai|artificial intelligence|nlp|computer vision)\b'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, jd_lower, re.IGNORECASE)
            for match in matches:
                skills = re.split(r'[,&/]', match)
                found_skills.extend([skill.strip() for skill in skills if skill.strip()])
        
        return list(set(found_skills))
    
    def analyze_resume_profile(self, resume_text, jd_skills):
        """Analyze resume against job description skills"""
        if not resume_text or not jd_skills:
            return {
                'matched_skills': [],
                'missing_skills': jd_skills,
                'match_percentage': 0,
                'total_jd_skills': len(jd_skills),
                'skills_found': 0
            }
            
        resume_lower = resume_text.lower()
        
        matched_skills = []
        missing_skills = []
        
        for skill in jd_skills:
            if skill.lower() in resume_lower:
                matched_skills.append(skill)
            else:
                missing_skills.append(skill)
        
        match_percentage = (len(matched_skills) / len(jd_skills)) * 100 if jd_skills else 0
        
        return {
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'match_percentage': match_percentage,
            'total_jd_skills': len(jd_skills),
            'skills_found': len(matched_skills)
        }
    
    def generate_resume_from_scratch(self, job_title, job_description, user_info=None, ollama_ai=None):
        """Generate a complete resume from scratch using Ollama AI - MANDATORY"""
        if user_info is None:
            user_info = {}
        
        # Use Ollama to generate unique resume content
        if ollama_ai:
            try:
                resume = ollama_ai.generate_complete_resume(user_info, job_title, job_description)
                if resume:
                    return resume.strip()
                else:
                    st.warning("Ollama generation failed, using fallback method.")
            except Exception as e:
                st.error(f"Error using Ollama: {str(e)}")
                raise
        
        # Fallback (should not be reached if Ollama is mandatory)
        jd_skills = self.extract_skills_from_jd(job_description)
        name = user_info.get('name', 'Your Name')
        email = user_info.get('email', 'your.email@example.com')
        phone = user_info.get('phone', '(123) 456-7890')
        location = user_info.get('location', 'City, State')
        education = user_info.get('education', 'Bachelor of Science in Relevant Field')
        years_experience = user_info.get('years_experience', 3)
        
        resume = f"""{name.upper()}
{email} | {phone} | {location}

PROFESSIONAL SUMMARY
Results-driven {job_title} with {years_experience}+ years of experience.

TECHNICAL SKILLS
{', '.join(jd_skills[:15]) if jd_skills else 'Relevant technical skills'}

PROFESSIONAL EXPERIENCE
{job_title} | Company Name | {location} | {datetime.now().year - years_experience} - Present
• Relevant experience and achievements

EDUCATION
{education} | University Name | {datetime.now().year - years_experience - 4}
"""
        return resume.strip()
    
    def _generate_experience_bullets(self, jd_skills, job_title, previous=False):
        """Generate relevant experience bullet points"""
        bullets = []
        num_bullets = 4 if not previous else 3
        
        experience_templates = [
            "• {action} {skill} solutions {impact}",
            "• {action} and maintained {skill} systems {impact}",
            "• Led {skill} initiatives that {impact}",
            "• Collaborated on cross-functional teams to {action} {skill} applications {impact}",
            "• {action} scalable {skill} architectures {impact}",
            "• Optimized {skill} processes {impact}",
            "• Implemented {skill} features that {impact}"
        ]
        
        for i in range(num_bullets):
            action = random.choice(self.action_verbs)
            skill = random.choice(jd_skills) if jd_skills else "technology"
            impact = random.choice(self.impact_phrases)
            
            template = random.choice(experience_templates)
            bullet = template.format(action=action.capitalize(), skill=skill, impact=impact)
            bullets.append(bullet)
        
        return '\n'.join(bullets)
    
    def _generate_projects(self, jd_skills):
        """Generate relevant project descriptions"""
        if not jd_skills:
            jd_skills = ['Python', 'JavaScript', 'SQL', 'React']
            
        projects = []
        
        project1 = f"""E-Commerce Platform | {datetime.now().year - 1}
• Developed a full-stack application using {jd_skills[0] if len(jd_skills) > 0 else 'Python'} and {jd_skills[1] if len(jd_skills) > 1 else 'React'} {random.choice(self.impact_phrases)}
• Implemented user authentication, product catalog, and shopping cart functionality
• Utilized {jd_skills[2] if len(jd_skills) > 2 else 'MongoDB'} for data storage and deployed using Docker"""

        project2 = f"""Data Analysis Dashboard | {datetime.now().year}
• Built interactive dashboards using {jd_skills[3] if len(jd_skills) > 3 else 'Tableau'} to visualize key business metrics
• Processed large datasets with {jd_skills[0] if len(jd_skills) > 0 else 'Python'} and Pandas {random.choice(self.impact_phrases)}
• Integrated with REST APIs to provide real-time data updates and insights"""

        projects.extend([project1, project2])
        return '\n\n'.join(projects)
    
    def tailor_existing_resume(self, resume_text, job_title, job_description, jd_skills, ollama_ai):
        """Tailor an existing resume using Ollama AI - MANDATORY"""
        if not resume_text:
            return "No resume content provided."
        
        # Use Ollama to completely rewrite the resume
        if ollama_ai:
            try:
                tailored = ollama_ai.tailor_resume(resume_text, job_description, job_title)
                if tailored:
                    return tailored
                else:
                    st.warning("Ollama tailoring failed, using fallback method.")
            except Exception as e:
                st.error(f"Error using Ollama: {str(e)}")
                raise
        
        # Fallback (should not be reached if Ollama is mandatory)
        return resume_text

# ============================================================================
# AI INTEGRATION WITH OLLAMA
# ============================================================================

class OllamaAI:
    """AI integration using Ollama for resume enhancement - MANDATORY"""
    
    def __init__(self, base_url="http://localhost:11434", model="llama2"):
        self.base_url = base_url
        self.model = model
        self.available = OLLAMA_AVAILABLE
        self._connection_checked = False
        self._is_connected = False
    
    def check_connection(self):
        """Check if Ollama is running - MANDATORY"""
        if not self.available:
            return False
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            self._is_connected = response.status_code == 200
            self._connection_checked = True
            return self._is_connected
        except Exception as e:
            self._is_connected = False
            self._connection_checked = True
            return False
    
    def require_connection(self):
        """Require Ollama connection - raise error if not connected"""
        if not self._connection_checked:
            self.check_connection()
        if not self._is_connected:
            raise ConnectionError(
                f"Ollama is not running or not accessible at {self.base_url}. "
                f"Please start Ollama and ensure it's running on {self.base_url}. "
                f"Install Ollama from https://ollama.ai and run: ollama serve"
            )
        return True
    
    def generate_response(self, prompt, max_tokens=2000, temperature=0.8):
        """Generate AI response using Ollama - MANDATORY"""
        self.require_connection()
        
        if not self.available:
            raise RuntimeError("Ollama is not available. Install requests: pip install requests")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1  # Reduce repetition
                    }
                },
                timeout=120  # Longer timeout for larger responses
            )
            if response.status_code == 200:
                result = response.json().get("response", "")
                # Clean up the response
                if result:
                    # Remove any leading/trailing whitespace
                    result = result.strip()
                    # Remove common prefixes that models sometimes add
                    prefixes_to_remove = ["Here's", "Here is", "The tailored resume:", "Tailored Resume:", "Resume:"]
                    for prefix in prefixes_to_remove:
                        if result.startswith(prefix):
                            result = result[len(prefix):].strip()
                return result
            else:
                st.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.Timeout:
            st.error("Ollama request timed out. Try reducing the content size or using a faster model.")
            return None
        except Exception as e:
            st.error(f"Ollama error: {str(e)}")
            return None
    
    def enhance_resume_section(self, section_text, job_description=""):
        """Enhance a resume section using AI"""
        prompt = f"""You are a professional resume writer. Enhance the following resume section to make it more impactful and ATS-friendly.

Resume Section:
{section_text}

{f'Target Job Description: {job_description}' if job_description else ''}

Provide an improved version that:
1. Uses strong action verbs
2. Includes quantifiable achievements
3. Is concise and impactful
4. Matches ATS keywords

Enhanced version:"""
        
        return self.generate_response(prompt, max_tokens=300)
    
    def generate_professional_summary(self, resume_text, job_title, job_description=""):
        """Generate a professional summary using AI"""
        prompt = f"""Create a compelling professional summary for a resume.

Job Title: {job_title}
{f'Job Description: {job_description}' if job_description else ''}
Resume Content: {resume_text[:500]}

Write a 3-4 sentence professional summary that:
1. Highlights key qualifications
2. Mentions years of experience if available
3. Includes relevant skills
4. Is tailored to the job title

Professional Summary:"""
        
        return self.generate_response(prompt, max_tokens=200)
    
    def suggest_improvements(self, resume_text, job_description=""):
        """Get AI suggestions for resume improvements"""
        prompt = f"""Analyze this resume and provide specific improvement suggestions.

Resume:
{resume_text[:1000]}

{f'Target Job: {job_description}' if job_description else ''}

Provide 5-7 specific, actionable suggestions to improve this resume. Format as a numbered list.

Suggestions:"""
        
        return self.generate_response(prompt, max_tokens=400)
    
    def tailor_resume(self, resume_text, job_description, job_title=""):
        """Tailor resume to job description using AI - completely rewrite content"""
        # Use full resume text, not truncated
        full_resume = resume_text[:3000] if len(resume_text) > 3000 else resume_text
        
        prompt = f"""You are an expert resume writer. Completely tailor and rewrite this resume to match the job description. DO NOT just change job titles - rewrite the actual content, bullet points, and descriptions to align with the job requirements.

IMPORTANT INSTRUCTIONS:
1. Read the ENTIRE original resume carefully
2. Identify the person's actual experience and skills
3. Rewrite ALL sections (summary, experience bullets, skills) to match the job description
4. Use DIFFERENT wording and phrasing - avoid repetitive keywords
5. Make each bullet point unique and specific
6. Incorporate relevant keywords from the job description naturally
7. Maintain the person's actual experience but reframe it for this role
8. Use varied action verbs and impact statements
9. Keep the same structure but rewrite all content

Original Resume:
{full_resume}

Target Job Title: {job_title}

Job Description:
{job_description}

Now provide a COMPLETE rewritten resume that:
- Has been fully tailored to this specific job
- Uses varied, non-repetitive language
- Incorporates job-relevant keywords naturally
- Rewrites experience bullets to match job requirements
- Maintains authenticity while optimizing for ATS

Provide the complete tailored resume:"""
        
        return self.generate_response(prompt, max_tokens=2500, temperature=0.85)
    
    def generate_complete_resume(self, user_info, job_title, job_description):
        """Generate a complete resume from scratch using AI"""
        prompt = f"""You are an expert resume writer. Create a complete, professional resume for the following person targeting this specific job.

User Information:
- Name: {user_info.get('name', 'Candidate')}
- Email: {user_info.get('email', 'email@example.com')}
- Phone: {user_info.get('phone', '')}
- Location: {user_info.get('location', '')}
- Education: {user_info.get('education', '')}
- Years of Experience: {user_info.get('years_experience', 0)}

Target Job Title: {job_title}

Job Description:
{job_description}

Create a complete resume with:
1. Header with name, contact info
2. Professional Summary (3-4 sentences tailored to this job)
3. Technical Skills section (relevant to job description)
4. Professional Experience section (2-3 roles with 4-5 bullet points each)
   - Use varied action verbs
   - Include quantifiable achievements
   - Make each bullet unique
   - Avoid repetitive phrases
5. Education section
6. Projects section (2-3 relevant projects)
7. Certifications (if applicable)

IMPORTANT:
- Use DIFFERENT wording throughout - avoid repetition
- Make content specific to this job description
- Use varied action verbs (developed, implemented, architected, optimized, etc.)
- Include specific technologies and tools from job description
- Make each section unique and compelling

Format the resume with clear section headers in ALL CAPS. Provide the complete resume:"""
        
        return self.generate_response(prompt, max_tokens=2500, temperature=0.85)
    
    def enhance_experience_section(self, experience_text, job_description, job_title):
        """Enhance experience section with AI - rewrite bullets"""
        prompt = f"""Rewrite and enhance this experience section to match the job description. DO NOT just add keywords - completely rewrite the bullet points to be more relevant and impactful.

Original Experience Section:
{experience_text}

Target Job: {job_title}

Job Description:
{job_description}

Rewrite this experience section with:
- 4-5 new bullet points that match the job requirements
- Varied action verbs and phrasing
- Quantifiable achievements where possible
- Natural incorporation of relevant keywords
- Unique, non-repetitive content

Enhanced Experience Section:"""
        
        return self.generate_response(prompt, max_tokens=800, temperature=0.8)

# ============================================================================
# RESUME TEMPLATES
# ============================================================================

class ResumeTemplate:
    """Base class for resume templates"""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    def parse_resume_text(self, resume_text):
        """Parse resume text into structured sections - improved parsing"""
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
        header_processed = False
        
        for i, line in enumerate(lines):
            original_line = line
            line = line.strip()
            if not line:
                continue
            
            line_upper = line.upper()
            
            # Detect section headers more reliably
            if 'PROFESSIONAL SUMMARY' in line_upper or ('SUMMARY' in line_upper and 'PROFESSIONAL' in line_upper):
                current_section = 'summary'
                header_processed = True
                continue
            elif 'EXPERIENCE' in line_upper and ('PROFESSIONAL' in line_upper or 'WORK' in line_upper or 'EMPLOYMENT' in line_upper):
                current_section = 'experience'
                header_processed = True
                continue
            elif 'EDUCATION' in line_upper:
                current_section = 'education'
                header_processed = True
                continue
            elif ('SKILLS' in line_upper or 'TECHNICAL SKILLS' in line_upper) and 'EXPERIENCE' not in line_upper:
                current_section = 'skills'
                header_processed = True
                continue
            elif 'PROJECTS' in line_upper:
                current_section = 'projects'
                header_processed = True
                continue
            elif 'CERTIFICATIONS' in line_upper or 'CERTIFICATES' in line_upper:
                current_section = 'certifications'
                header_processed = True
                continue
            elif 'AWARDS' in line_upper or 'ACHIEVEMENTS' in line_upper:
                current_section = 'awards'
                header_processed = True
                continue
            
            # Parse header (first few lines before sections)
            if not header_processed and current_section is None:
                # Name is usually the first non-empty line
                if not sections['header']['name'] and len(line) < 60 and not any(c in line for c in ['|', '@', 'http', 'www']):
                    if line.isupper() or (len(line.split()) <= 4 and line[0].isupper()):
                        sections['header']['name'] = line
                        continue
                
                # Email
                if '@' in line and '.' in line:
                    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', line)
                    if email_match:
                        sections['header']['email'] = email_match.group()
                
                # Phone
                phone_match = re.search(r'[\d\s\-\(\)]{10,}', line)
                if phone_match and len(re.sub(r'[\s\-\(\)]', '', phone_match.group())) >= 10:
                    sections['header']['phone'] = phone_match.group().strip()
                
                # LinkedIn
                if 'linkedin.com' in line.lower():
                    linkedin_match = re.search(r'linkedin\.com/[^\s]+', line.lower())
                    if linkedin_match:
                        sections['header']['linkedin'] = 'https://' + linkedin_match.group()
                
                # Location (usually in contact line)
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    for part in parts:
                        if '@' in part:
                            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', part)
                            if email_match:
                                sections['header']['email'] = email_match.group()
                        elif re.match(r'[\d\s\-\(\)]{10,}', part):
                            sections['header']['phone'] = part
                        elif 'linkedin' in part.lower():
                            linkedin_match = re.search(r'linkedin\.com/[^\s]+', part.lower())
                            if linkedin_match:
                                sections['header']['linkedin'] = 'https://' + linkedin_match.group()
                        elif len(part) > 3 and not any(c in part for c in ['@', 'http', 'www']):
                            sections['header']['location'] = part
                
            # Parse section content
            elif current_section == 'summary':
                if sections['summary']:
                    sections['summary'] += ' ' + line
                else:
                    sections['summary'] = line
                    
            elif current_section == 'experience':
                # Job title line (usually has company and dates)
                if '|' in line or ' - ' in line or (i < len(lines) - 1 and not lines[i+1].strip().startswith(('•', '-', '*'))):
                    # This might be a job title line
                    parts = re.split(r'\s*\|\s*|\s*-\s*', line)
                    if len(parts) >= 2:
                        # Check if this looks like a job title (not a bullet)
                        if not line.startswith(('•', '-', '*')):
                            if current_item and current_item.get('title'):
                                sections['experience'].append(current_item)
                            current_item = {
                                'title': parts[0].strip(),
                                'company': parts[1].strip() if len(parts) > 1 else '',
                                'dates': parts[2].strip() if len(parts) > 2 else '',
                                'bullets': []
                            }
                            continue
                
                # Bullet points
                if line.startswith(('•', '-', '*')) or (current_item and line.strip()):
                    bullet_text = line.lstrip('•-*').strip()
                    if bullet_text:
                        if 'bullets' not in current_item:
                            current_item['bullets'] = []
                        current_item['bullets'].append(bullet_text)
                        current_item['description'] = '\n'.join(current_item['bullets'])
                        
            elif current_section == 'skills':
                if ',' in line:
                    skills_list = [s.strip() for s in line.split(',')]
                    sections['skills'].extend([s for s in skills_list if s])
                elif line.startswith(('•', '-', '*')):
                    skill = line.lstrip('•-*').strip()
                    if skill:
                        sections['skills'].append(skill)
                elif line and not line.isupper():  # Not a section header
                    sections['skills'].append(line)
                    
            elif current_section == 'education':
                sections['education'].append(line)
                
            elif current_section == 'projects':
                sections['projects'].append(line)
                
            elif current_section == 'certifications':
                sections['certifications'].append(line)
                
            elif current_section == 'awards':
                sections['awards'].append(line)
        
        # Add last experience item
        if current_item and current_item.get('title'):
            sections['experience'].append(current_item)
        
        # Ensure we have at least the name from first line if not parsed
        if not sections['header']['name'] and lines:
            first_line = lines[0].strip()
            if first_line and len(first_line) < 60:
                sections['header']['name'] = first_line
        
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
        
        header_style = ParagraphStyle('Header', parent=styles['Normal'],
            fontSize=24, textColor=colors.HexColor('#2c3e50'),
            spaceAfter=6, fontName='Helvetica-Bold', alignment=TA_CENTER)
        
        contact_style = ParagraphStyle('Contact', parent=styles['Normal'],
            fontSize=10, textColor=colors.HexColor('#34495e'),
            spaceAfter=12, alignment=TA_CENTER)
        
        section_style = ParagraphStyle('Section', parent=styles['Heading2'],
            fontSize=14, textColor=colors.HexColor('#3498db'),
            spaceAfter=8, spaceBefore=12, fontName='Helvetica-Bold')
        
        normal_style = ParagraphStyle('Normal', parent=styles['Normal'],
            fontSize=10, textColor=colors.black, spaceAfter=4, leading=12)
        
        bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'],
            fontSize=10, textColor=colors.black, spaceAfter=3,
            leftIndent=15, bulletIndent=5)
        
        if sections['header']['name']:
            story.append(Paragraph(sections['header']['name'].upper(), header_style))
        
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
        
        if sections['summary']:
            story.append(Paragraph("PROFESSIONAL SUMMARY", section_style))
            story.append(Paragraph(sections['summary'], normal_style))
            story.append(Spacer(1, 0.15*inch))
        
        if sections['experience']:
            story.append(Paragraph("PROFESSIONAL EXPERIENCE", section_style))
            for exp in sections['experience'][:5]:
                if exp.get('title'):
                    title_text = f"<b>{exp['title']}</b>"
                    if exp.get('company'):
                        title_text += f" | {exp['company']}"
                    if exp.get('dates'):
                        title_text += f" | {exp['dates']}"
                    story.append(Paragraph(title_text, normal_style))
                    story.append(Spacer(1, 0.05*inch))
                
                # Handle bullets if they exist
                if exp.get('bullets'):
                    for bullet in exp['bullets']:
                        if bullet.strip():
                            story.append(Paragraph(f"• {bullet.strip()}", bullet_style))
                elif exp.get('description'):
                    # Split description by newlines or bullets
                    desc_lines = exp['description'].replace('•', '\n').replace('-', '\n').split('\n')
                    for desc_line in desc_lines:
                        desc_line = desc_line.strip()
                        if desc_line:
                            story.append(Paragraph(f"• {desc_line}", bullet_style))
                
                story.append(Spacer(1, 0.1*inch))
        
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
        
        name_style = ParagraphStyle('Name', parent=styles['Heading1'],
            fontSize=18, textColor=colors.black, spaceAfter=6,
            fontName='Helvetica-Bold', alignment=TA_CENTER)
        
        contact_style = ParagraphStyle('Contact', parent=styles['Normal'],
            fontSize=10, textColor=colors.black, spaceAfter=15, alignment=TA_CENTER)
        
        section_style = ParagraphStyle('Section', parent=styles['Heading2'],
            fontSize=12, textColor=colors.black, spaceAfter=6,
            spaceBefore=10, fontName='Helvetica-Bold',
            borderWidth=1, borderPadding=2, borderColor=colors.black)
        
        normal_style = ParagraphStyle('Normal', parent=styles['Normal'],
            fontSize=10, textColor=colors.black, spaceAfter=4, leading=12)
        
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
                    story.append(Spacer(1, 0.05*inch))
                
                # Handle bullets if they exist
                if exp.get('bullets'):
                    for bullet in exp['bullets']:
                        if bullet.strip():
                            story.append(Paragraph(f"• {bullet.strip()}", normal_style))
                elif exp.get('description'):
                    # Split description by newlines or bullets
                    desc_lines = exp['description'].replace('•', '\n').replace('-', '\n').split('\n')
                    for desc_line in desc_lines:
                        desc_line = desc_line.strip()
                        if desc_line:
                            story.append(Paragraph(f"• {desc_line}", normal_style))
                
                story.append(Spacer(1, 0.08*inch))
        
        if sections['skills']:
            story.append(Paragraph("SKILLS", section_style))
            skills_text = ', '.join(sections['skills'][:20])
            story.append(Paragraph(skills_text, normal_style))
        
        doc.build(story)
        buffer.seek(0)
        return buffer

# ============================================================================
# PDF GENERATION WITH TEMPLATE SUPPORT
# ============================================================================

def generate_pdf(resume_text, template_name="Default"):
    """Generate PDF using reportlab with proper binary output"""
    try:
        # Validate input
        if not resume_text or not resume_text.strip():
            raise Exception("Resume text is empty")
        
        # Use template if specified
        if template_name in ["Modern", "Classic"]:
            templates = {
                "Modern": ModernTemplate(),
                "Classic": ClassicTemplate()
            }
            return templates[template_name].generate_pdf(resume_text)
        
        # Default template (original behavior)
        buffer = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        story = []
        
        # Define styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor='black',
            spaceAfter=12,
            alignment=TA_LEFT
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor='black',
            spaceAfter=8,
            spaceBefore=12,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            textColor='black',
            spaceAfter=6,
            alignment=TA_LEFT,
            leftIndent=0
        )
        
        bullet_style = ParagraphStyle(
            'CustomBullet',
            parent=styles['Normal'],
            fontSize=11,
            textColor='black',
            spaceAfter=5,
            alignment=TA_LEFT,
            leftIndent=20,
            bulletIndent=10
        )
        
        # Process resume text
        lines = resume_text.split('\n')
        content_added = False
        
        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 6))
                continue
            
            content_added = True
            
            # Handle different content types
            if line.isupper() and len(line) < 100 and not line.startswith(('-', '*', '•')):  # Headers
                # Section headers
                story.append(Paragraph(line, heading_style))
                story.append(Spacer(1, 6))
            elif line.startswith('•') or line.startswith('-') or line.startswith('*'):  # Bullet points
                # Remove bullet and add as paragraph
                clean_line = line.lstrip('•-*').strip()
                if clean_line:
                    # Format as bullet point
                    bullet_text = f"• {clean_line}"
                    story.append(Paragraph(bullet_text, bullet_style))
            else:  # Regular text
                # Escape special characters for reportlab
                escaped_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(escaped_line, normal_style))
        
        # Ensure we added content
        if not content_added:
            story.append(Paragraph("Resume Content", normal_style))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF bytes from buffer
        buffer.seek(0)
        pdf_bytes = buffer.getvalue()
        
        # Validate PDF
        if len(pdf_bytes) < 4:
            raise Exception(f"PDF output is too short: {len(pdf_bytes)} bytes")
        
        if pdf_bytes[:4] != b'%PDF':
            raise Exception(f"Invalid PDF header. First 30 bytes: {pdf_bytes[:30]}")
        
        # Create fresh buffer with PDF bytes
        result_buffer = io.BytesIO(pdf_bytes)
        result_buffer.seek(0)
        
        return result_buffer
        
    except Exception as e:
        error_msg = f"PDF Error: {str(e)}"
        st.error(error_msg)
        # Return text file as fallback
        return generate_text_file(resume_text)

def generate_text_file(resume_text):
    """Generate text file as fallback"""
    buffer = io.BytesIO()
    buffer.write(resume_text.encode('utf-8'))
    buffer.seek(0)
    return buffer

def display_pdf_viewer(pdf_bytes):
    """Display PDF preview in the app"""
    try:
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not display PDF preview: {str(e)}")

def main():
    st.set_page_config(
        page_title="AI Resume Generator with PDF Editing",
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
    .upload-box {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🤖 AI Resume Generator</h1>', unsafe_allow_html=True)
    st.markdown("**Upload, Edit, and Generate Resumes with PDF Support**")
    
    # Initialize processor
    processor = ResumeProcessor()
    
    # Initialize session state
    if 'resume_content' not in st.session_state:
        st.session_state.resume_content = ""
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'edited_content' not in st.session_state:
        st.session_state.edited_content = ""
    if 'resume_option' not in st.session_state:
        st.session_state.resume_option = "upload"

    # Initialize AI - MANDATORY
    if 'ollama' not in st.session_state:
        st.session_state.ollama = OllamaAI()
    
    # Check Ollama connection - MANDATORY
    with st.spinner("Checking Ollama connection..."):
        if not st.session_state.ollama.check_connection():
            st.error("""
            ## ❌ Ollama is Required!
            
            This application **requires Ollama** to be running locally.
            
            **To fix this:**
            1. Install Ollama from https://ollama.ai
            2. Start Ollama service (it usually runs automatically)
            3. Pull a model: `ollama pull llama2` (or another model)
            4. Refresh this page
            
            **Current Ollama URL:** {}
            """.format(st.session_state.ollama.base_url))
            st.stop()
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("🚀 Navigation")
        option = st.radio(
            "Choose your action:",
            ["📤 Upload & Edit PDF", "🎯 Generate from Scratch", "🔧 Tailor Existing Resume"]
        )
        
        st.header("🎨 Template Selection")
        template_option = st.selectbox(
            "Choose Template",
            ["Default", "Modern", "Classic"],
            help="Select a resume template style"
        )
        
        st.header("🤖 Ollama Settings (Required)")
        st.info("✅ Ollama is connected and required for all operations")
        ollama_url = st.text_input("Ollama URL", st.session_state.ollama.base_url, help="Ollama server URL")
        ollama_model = st.text_input("Model", st.session_state.ollama.model, help="Ollama model name (e.g., llama2, mistral, codellama)")
        st.session_state.ollama.base_url = ollama_url
        st.session_state.ollama.model = ollama_model
        
        if st.button("🔌 Test Connection"):
            if st.session_state.ollama.check_connection():
                st.success("✅ Connected to Ollama!")
            else:
                st.error("❌ Cannot connect to Ollama. Make sure it's running.")
        
        st.header("⚙️ Settings")
        job_title = st.text_input("Target Job Title", "Software Engineer")
        job_description = st.text_area("Job Description", height=150)
    
    # Main content area
    if option == "📤 Upload & Edit PDF":
        st.header("📤 Upload & Edit Your PDF Resume")
        
        # File upload section
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag and drop your PDF resume here",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            
            with st.spinner("Extracting text from your document..."):
                try:
                    if uploaded_file.type == "application/pdf":
                        extracted_text = processor.extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        extracted_text = processor.extract_text_from_docx(uploaded_file)
                    else:
                        extracted_text = uploaded_file.getvalue().decode("utf-8")
                    
                    if extracted_text:
                        st.session_state.resume_content = extracted_text
                        st.session_state.edited_content = extracted_text
                        st.success(f"✅ Successfully extracted {len(extracted_text)} characters!")
                    else:
                        st.error("Could not extract text from the document.")
                        
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
            
            # Display extracted content for editing
            if st.session_state.resume_content:
                # Analysis and Enhancement Section
                if job_description:
                    st.markdown("---")
                    st.subheader("📊 Resume Analysis & ATS Score")
                    
                    with st.spinner("Analyzing your resume..."):
                        jd_skills = processor.extract_skills_from_jd(job_description)
                        analysis = processor.analyze_resume_profile(st.session_state.resume_content, jd_skills)
                        
                        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
                        with col_a1:
                            st.metric("ATS Match Score", f"{analysis['match_percentage']:.1f}%", 
                                    delta=f"{analysis['match_percentage'] - 50:.1f}%" if analysis['match_percentage'] > 50 else None)
                        with col_a2:
                            st.metric("Skills Matched", f"{analysis['skills_found']}/{analysis['total_jd_skills']}")
                        with col_a3:
                            st.metric("Missing Skills", len(analysis['missing_skills']))
                        with col_a4:
                            word_count = len(st.session_state.resume_content.split())
                            st.metric("Word Count", word_count)
                    
                    # Detailed Analysis Tabs
                    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["✅ Matched Skills", "❌ Missing Skills", "💡 Suggestions"])
                    
                    with analysis_tab1:
                        if analysis['matched_skills']:
                            st.success(f"**Found {len(analysis['matched_skills'])} matching skills:**")
                            st.write(", ".join(analysis['matched_skills']))
                        else:
                            st.warning("No matching skills found. Consider adding relevant skills from the job description.")
                    
                    with analysis_tab2:
                        if analysis['missing_skills']:
                            st.warning(f"**Missing {len(analysis['missing_skills'])} important skills:**")
                            missing_display = ", ".join(analysis['missing_skills'][:15])
                            st.write(missing_display)
                            if len(analysis['missing_skills']) > 15:
                                st.caption(f"... and {len(analysis['missing_skills']) - 15} more")
                            
                            if st.button("➕ Add Top Missing Skills to Resume", key="add_missing_skills"):
                                lines = st.session_state.edited_content.split('\n')
                                skills_to_add = analysis['missing_skills'][:5]
                                
                                # Find or create skills section
                                skills_section_idx = None
                                for i, line in enumerate(lines):
                                    if "SKILL" in line.upper() and "SECTION" not in line.upper():
                                        skills_section_idx = i
                                        break
                                
                                if skills_section_idx is not None:
                                    # Add to existing skills line
                                    if skills_section_idx + 1 < len(lines):
                                        existing_skills = lines[skills_section_idx + 1]
                                        new_skills_str = ", ".join(skills_to_add)
                                        lines[skills_section_idx + 1] = existing_skills + f", {new_skills_str}"
                                    else:
                                        lines.append(", ".join(skills_to_add))
                                else:
                                    # Create new skills section
                                    lines.append("\nSKILLS")
                                    lines.append(", ".join(skills_to_add))
                                
                                st.session_state.edited_content = '\n'.join(lines)
                                st.success(f"Added {len(skills_to_add)} missing skills!")
                                st.rerun()
                        else:
                            st.success("Great! You have all the required skills mentioned in the job description.")
                    
                    with analysis_tab3:
                        st.info("**Enhancement Suggestions:**")
                        suggestions = []
                        
                        # Check for professional summary
                        if "SUMMARY" not in st.session_state.resume_content.upper() and "OBJECTIVE" not in st.session_state.resume_content.upper():
                            suggestions.append("📝 Add a Professional Summary section to highlight your key qualifications")
                        
                        # Check for quantified achievements
                        has_numbers = any(char.isdigit() for char in st.session_state.resume_content)
                        if not has_numbers:
                            suggestions.append("📊 Add quantified achievements (e.g., 'increased sales by 20%', 'managed team of 5')")
                        
                        # Check for action verbs
                        action_verb_count = sum(1 for verb in processor.action_verbs if verb in st.session_state.resume_content.lower())
                        if action_verb_count < 5:
                            suggestions.append("⚡ Use more action verbs (e.g., 'developed', 'implemented', 'led', 'optimized')")
                        
                        # Check resume length
                        if word_count < 300:
                            suggestions.append("📏 Resume seems short. Consider adding more details about your experience")
                        elif word_count > 800:
                            suggestions.append("📏 Resume is quite long. Consider condensing to 1-2 pages")
                        
                        if suggestions:
                            for suggestion in suggestions:
                                st.write(f"• {suggestion}")
                        else:
                            st.success("Your resume looks well-structured!")
                        
                        # AI Suggestions - Always available
                        st.markdown("---")
                        st.subheader("🤖 AI-Powered Suggestions")
                        if st.button("✨ Get AI Suggestions", key="ai_suggestions"):
                            with st.spinner("AI is analyzing your resume..."):
                                try:
                                    ai_suggestions = st.session_state.ollama.suggest_improvements(
                                        st.session_state.edited_content, job_description
                                    )
                                    if ai_suggestions:
                                        st.info(ai_suggestions)
                                    else:
                                        st.warning("Could not get AI suggestions.")
                                except Exception as e:
                                    st.error(f"Error getting AI suggestions: {str(e)}")
                
                st.markdown("---")
                st.subheader("✏️ Edit Your Resume Content")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Content (Read-only)**")
                    st.text_area("Original", st.session_state.resume_content, height=400, key="original_view")
                    
                    # Text Statistics
                    with st.expander("📈 Text Statistics"):
                        stats_text = st.session_state.edited_content
                        word_count = len(stats_text.split())
                        char_count = len(stats_text)
                        line_count = len(stats_text.split('\n'))
                        bullet_count = stats_text.count('•') + stats_text.count('-')
                        
                        st.metric("Words", word_count)
                        st.metric("Characters", char_count)
                        st.metric("Lines", line_count)
                        st.metric("Bullet Points", bullet_count)
                
                with col2:
                    st.write("**Editable Content**")
                    st.session_state.edited_content = st.text_area(
                        "Make your edits here", 
                        st.session_state.edited_content, 
                        height=400,
                        key="editable_content"
                    )
                    
                    # Enhanced Edit tools
                    st.subheader("🛠️ Quick Edit Tools")
                    
                    col_t1, col_t2 = st.columns(2)
                    
                    with col_t1:
                        if st.button("🔄 Reset Changes"):
                            st.session_state.edited_content = st.session_state.resume_content
                            st.rerun()
                        
                        if st.button("📝 Add Professional Summary"):
                            if "SUMMARY" not in st.session_state.edited_content.upper() and "OBJECTIVE" not in st.session_state.edited_content.upper():
                                # Use AI to generate summary - MANDATORY
                                with st.spinner("AI is generating professional summary..."):
                                    try:
                                        ai_summary = st.session_state.ollama.generate_professional_summary(
                                            st.session_state.edited_content, job_title, job_description
                                        )
                                        if ai_summary:
                                            summary = f"\n\nPROFESSIONAL SUMMARY\n{ai_summary}"
                                            st.session_state.edited_content = summary + "\n\n" + st.session_state.edited_content
                                            st.success("Professional Summary added!")
                                        else:
                                            st.error("Could not generate summary. Please try again.")
                                    except Exception as e:
                                        st.error(f"Error generating summary: {str(e)}")
                            else:
                                st.warning("Summary section already exists!")
                        
                        # AI Enhancement button - Always available
                        if st.button("🤖 AI Enhance Resume", key="ai_enhance"):
                            with st.spinner("AI is enhancing your resume..."):
                                try:
                                    enhanced = st.session_state.ollama.enhance_resume_section(
                                        st.session_state.edited_content, job_description
                                    )
                                    if enhanced:
                                        st.session_state.edited_content = enhanced
                                        st.success("Resume enhanced with AI!")
                                        st.rerun()
                                    else:
                                        st.warning("Could not enhance resume.")
                                except Exception as e:
                                    st.error(f"Error enhancing resume: {str(e)}")
                        
                        if st.button("🔍 Highlight Skills in Resume"):
                            if job_description:
                                jd_skills = processor.extract_skills_from_jd(job_description)
                                resume_lower = st.session_state.edited_content.lower()
                                found_skills = [skill for skill in jd_skills if skill.lower() in resume_lower]
                                if found_skills:
                                    st.success(f"**Found {len(found_skills)} skills from job description:**")
                                    st.write(", ".join(found_skills[:10]))
                                    if len(found_skills) > 10:
                                        st.caption(f"... and {len(found_skills) - 10} more")
                                else:
                                    st.warning("No matching skills found in resume.")
                            else:
                                st.warning("Please enter a job description in the sidebar first.")
                    
                    with col_t2:
                        if st.button("🔧 Enhance Bullet Points"):
                            # Enhanced bullet point improvement
                            lines = st.session_state.edited_content.split('\n')
                            enhanced_lines = []
                            enhanced_count = 0
                            for line in lines:
                                if line.strip().startswith(('•', '-', '*')):
                                    if not any(phrase in line for phrase in processor.impact_phrases):
                                        if random.random() > 0.5:  # 50% chance to enhance
                                            line = line.rstrip('.') + f", {random.choice(processor.impact_phrases)}."
                                            enhanced_count += 1
                                enhanced_lines.append(line)
                            st.session_state.edited_content = '\n'.join(enhanced_lines)
                            if enhanced_count > 0:
                                st.success(f"Enhanced {enhanced_count} bullet points!")
                            else:
                                st.info("No bullet points needed enhancement.")
                        
                        if st.button("🎯 Add Missing Skills") and job_description:
                            jd_skills = processor.extract_skills_from_jd(job_description)
                            analysis = processor.analyze_resume_profile(st.session_state.edited_content, jd_skills)
                            if analysis['missing_skills']:
                                lines = st.session_state.edited_content.split('\n')
                                skills_to_add = analysis['missing_skills'][:5]
                                
                                # Find skills section
                                skills_found = False
                                for i, line in enumerate(lines):
                                    if "SKILL" in line.upper() and "SECTION" not in line.upper():
                                        if i + 1 < len(lines):
                                            lines[i + 1] = lines[i + 1] + f", {', '.join(skills_to_add)}"
                                        else:
                                            lines.append(", ".join(skills_to_add))
                                        skills_found = True
                                        break
                                
                                if not skills_found:
                                    lines.append("\nSKILLS")
                                    lines.append(", ".join(skills_to_add))
                                
                                st.session_state.edited_content = '\n'.join(lines)
                                st.success(f"Added {len(skills_to_add)} missing skills!")
                            else:
                                st.info("No missing skills to add!")
                        
                        if st.button("✨ Format Resume"):
                            # Basic formatting improvements
                            lines = st.session_state.edited_content.split('\n')
                            formatted_lines = []
                            for line in lines:
                                line = line.strip()
                                if line:
                                    # Ensure proper spacing for headers
                                    if line.isupper() and len(line) < 100:
                                        if formatted_lines and formatted_lines[-1] != "":
                                            formatted_lines.append("")
                                        formatted_lines.append(line)
                                        formatted_lines.append("")
                                    else:
                                        formatted_lines.append(line)
                                else:
                                    formatted_lines.append("")
                            
                            st.session_state.edited_content = '\n'.join(formatted_lines)
                            st.success("Resume formatted!")
                        
                        # AI Tailor button - Always available
                        if job_description:
                            if st.button("🎯 AI Tailor Resume", key="ai_tailor"):
                                with st.spinner("AI is tailoring your resume to the job description (this may take a minute)..."):
                                    try:
                                        tailored = st.session_state.ollama.tailor_resume(
                                            st.session_state.edited_content, job_description, job_title
                                        )
                                        if tailored:
                                            st.session_state.edited_content = tailored
                                            st.success("Resume tailored with AI!")
                                            st.rerun()
                                        else:
                                            st.warning("Could not tailor resume.")
                                    except Exception as e:
                                        st.error(f"Error tailoring resume: {str(e)}")
                        else:
                            st.info("Enter a job description in the sidebar to tailor your resume.")
                
                # Preview and Download Section
                st.markdown("---")
                st.header("👀 Preview & Download")
                
                preview_tab, download_tab = st.tabs(["📄 Preview", "💾 Download"])
                
                with preview_tab:
                    st.subheader("Resume Preview")
                    st.text_area("Current Content", st.session_state.edited_content, height=300)
                    
                    # Generate PDF preview
                    if st.button("Generate PDF Preview"):
                        with st.spinner("Creating PDF preview..."):
                            try:
                                pdf_buffer = generate_pdf(st.session_state.edited_content, template_option)
                                pdf_bytes = pdf_buffer.getvalue()
                                display_pdf_viewer(pdf_bytes)
                            except Exception as e:
                                st.error(f"Error generating PDF: {str(e)}")
                
                with download_tab:
                    st.subheader("Download Options")
                    
                    col_d1, col_d2 = st.columns(2)
                    
                    with col_d1:
                        # PDF Download - FIXED: Get bytes from buffer
                        try:
                            pdf_buffer = generate_pdf(st.session_state.edited_content, template_option)
                            pdf_bytes = pdf_buffer.getvalue()
                            st.download_button(
                                label="📄 Download as PDF",
                                data=pdf_bytes,
                                file_name=f"edited_resume_{template_option.lower()}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                    
                    with col_d2:
                        # Text Download
                        text_buffer = generate_text_file(st.session_state.edited_content)
                        st.download_button(
                            label="📝 Download as Text",
                            data=text_buffer,
                            file_name="edited_resume.txt",
                            mime="text/plain",
                            use_container_width=True
                        )

    elif option == "🎯 Generate from Scratch":
        st.header("🎯 Generate Resume from Scratch")
        
        st.subheader("👤 Your Information")
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name", "John Doe")
            email = st.text_input("Email", "john.doe@email.com")
            phone = st.text_input("Phone", "(123) 456-7890")
        
        with col2:
            location = st.text_input("Location", "San Francisco, CA")
            education = st.text_input("Education", "Bachelor of Science in Computer Science")
            years_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=3)
        
        if st.button("🚀 Generate Resume", type="primary", use_container_width=True):
            if not job_title or not job_description:
                st.error("Please enter both Job Title and Job Description")
            else:
                with st.spinner("AI is creating your professional resume (this may take a minute)..."):
                    try:
                        user_info = {
                            'name': name,
                            'email': email,
                            'phone': phone,
                            'location': location,
                            'education': education,
                            'years_experience': years_experience
                        }
                        
                        # Generate resume using Ollama - MANDATORY
                        resume = processor.generate_resume_from_scratch(
                            job_title, job_description, user_info, st.session_state.ollama
                        )
                        st.session_state.edited_content = resume
                        
                        st.balloons()
                        st.success("✅ Resume generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Display generated resume
        if st.session_state.edited_content and option == "🎯 Generate from Scratch":
            st.subheader("✨ Generated Resume")
            st.text_area("Generated Content", st.session_state.edited_content, height=400)
            
            # Download options
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                try:
                    pdf_buffer = generate_pdf(st.session_state.edited_content, template_option)
                    pdf_bytes = pdf_buffer.getvalue()
                    st.download_button(
                        label="📄 Download PDF",
                        data=pdf_bytes,
                        file_name=f"generated_resume_{template_option.lower()}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
            with col_d2:
                text_buffer = generate_text_file(st.session_state.edited_content)
                st.download_button(
                    label="📝 Download Text",
                    data=text_buffer,
                    file_name="generated_resume.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    elif option == "🔧 Tailor Existing Resume":
        st.header("🔧 Tailor Existing Resume")
        
        if not st.session_state.resume_content:
            st.warning("Please upload a resume first in the 'Upload & Edit PDF' section.")
        elif not job_description:
            st.warning("Please enter a job description in the sidebar.")
        else:
            st.subheader("Original Resume Content")
            st.text_area("Original", st.session_state.resume_content, height=200, key="tailor_original")
            
            if st.button("🎯 Tailor for Job Description", type="primary"):
                with st.spinner("AI is tailoring your resume (this may take a minute)..."):
                    try:
                        jd_skills = processor.extract_skills_from_jd(job_description)
                        tailored_resume = processor.tailor_existing_resume(
                            st.session_state.resume_content, 
                            job_title, 
                            job_description, 
                            jd_skills,
                            st.session_state.ollama
                        )
                        st.session_state.edited_content = tailored_resume
                        
                        # Show analysis
                        analysis = processor.analyze_resume_profile(tailored_resume, jd_skills)
                        
                        col_a1, col_a2 = st.columns(2)
                        with col_a1:
                            st.metric("Match Score", f"{analysis['match_percentage']:.1f}%")
                        with col_a2:
                            st.metric("Skills Matched", f"{analysis['skills_found']}/{analysis['total_jd_skills']}")
                        
                        st.success("✅ Resume tailored successfully!")
                    except Exception as e:
                        st.error(f"Error tailoring resume: {str(e)}")
            
            if st.session_state.edited_content and option == "🔧 Tailor Existing Resume":
                st.subheader("✨ Tailored Resume")
                st.text_area("Tailored Content", st.session_state.edited_content, height=400)
                
                # Download tailored resume
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    try:
                        pdf_buffer = generate_pdf(st.session_state.edited_content, template_option)
                        pdf_bytes = pdf_buffer.getvalue()
                        st.download_button(
                            label="📄 Download Tailored PDF",
                            data=pdf_bytes,
                            file_name=f"tailored_resume_{template_option.lower()}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")
                with col_d2:
                    text_buffer = generate_text_file(st.session_state.edited_content)
                    st.download_button(
                        label="📝 Download Tailored Text",
                        data=text_buffer,
                        file_name="tailored_resume.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

    # Footer
    st.markdown("---")
    st.markdown("*Upload, edit, and generate resumes with full PDF support!*")

if __name__ == "__main__":
    main()