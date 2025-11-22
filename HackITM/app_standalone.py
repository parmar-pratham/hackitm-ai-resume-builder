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
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

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
    
    def extract_ats_keywords(self, job_description):
        """Extract comprehensive ATS keywords from job description"""
        if not job_description:
            return {
                'skills': [],
                'technologies': [],
                'qualifications': [],
                'soft_skills': [],
                'certifications': [],
                'education_keywords': [],
                'experience_keywords': [],
                'all_keywords': []
            }
        
        jd_lower = job_description.lower()
        keywords = {
            'skills': [],
            'technologies': [],
            'qualifications': [],
            'soft_skills': [],
            'certifications': [],
            'education_keywords': [],
            'experience_keywords': [],
            'all_keywords': []
        }
        
        # Technical Skills
        technical_keywords = [
            'python', 'java', 'javascript', 'typescript', 'sql', 'html', 'css', 'react', 'angular', 'vue',
            'node.js', 'express', 'django', 'flask', 'spring', 'c++', 'c#', '.net', 'php', 'ruby', 'go',
            'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'bash', 'powershell', 'shell scripting'
        ]
        
        # Technologies & Tools
        tech_tools = [
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'git', 'github', 'gitlab',
            'ci/cd', 'devops', 'agile', 'scrum', 'jira', 'confluence', 'mongodb', 'postgresql', 'mysql',
            'redis', 'elasticsearch', 'kafka', 'rabbitmq', 'nginx', 'apache', 'linux', 'windows', 'macos',
            'tableau', 'power bi', 'excel', 'pandas', 'numpy', 'tensorflow', 'pytorch', 'keras', 'scikit-learn'
        ]
        
        # Soft Skills
        soft_skills_list = [
            'leadership', 'communication', 'teamwork', 'collaboration', 'problem solving', 'critical thinking',
            'analytical', 'creative', 'adaptable', 'time management', 'project management', 'presentation',
            'negotiation', 'mentoring', 'coaching', 'stakeholder management', 'client relations'
        ]
        
        # Qualifications & Requirements
        qualification_keywords = [
            'bachelor', 'master', 'phd', 'degree', 'certification', 'certified', 'years of experience',
            'minimum', 'required', 'preferred', 'must have', 'should have', 'proven track record'
        ]
        
        # Education Keywords
        education_keywords = [
            'computer science', 'engineering', 'information technology', 'data science', 'business',
            'mba', 'bachelor of science', 'master of science', 'phd', 'doctorate'
        ]
        
        # Experience Keywords
        experience_keywords = [
            'years of experience', 'yoe', 'senior', 'junior', 'mid-level', 'entry level', 'experienced',
            'expert', 'proficient', 'advanced', 'intermediate', 'beginner'
        ]
        
        # Extract keywords
        for keyword in technical_keywords:
            if keyword in jd_lower:
                keywords['skills'].append(keyword.title())
        
        for keyword in tech_tools:
            if keyword in jd_lower:
                keywords['technologies'].append(keyword.upper() if keyword.isupper() else keyword.title())
        
        for keyword in soft_skills_list:
            if keyword in jd_lower:
                keywords['soft_skills'].append(keyword.title())
        
        # Extract certifications
        cert_patterns = [
            r'\b(?:AWS|Azure|GCP|Google Cloud|Microsoft|Oracle|Cisco|PMP|Scrum|Agile)\s+(?:Certified|Certification|Certificate)',
            r'\b(?:Certified|Certification)\s+(?:in|for)\s+([A-Za-z\s]+)',
            r'\b([A-Z]{2,10})\s+Certification'
        ]
        for pattern in cert_patterns:
            matches = re.findall(pattern, job_description, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    keywords['certifications'].extend([m.strip() for m in match if m.strip()])
                else:
                    keywords['certifications'].append(match.strip())
        
        # Extract qualifications
        for keyword in qualification_keywords:
            if keyword in jd_lower:
                keywords['qualifications'].append(keyword.title())
        
        # Extract education keywords
        for keyword in education_keywords:
            if keyword in jd_lower:
                keywords['education_keywords'].append(keyword.title())
        
        # Extract experience keywords
        for keyword in experience_keywords:
            if keyword in jd_lower:
                keywords['experience_keywords'].append(keyword.title())
        
        # Also extract from skills list
        jd_skills = self.extract_skills_from_jd(job_description)
        keywords['skills'].extend([s for s in jd_skills if s.lower() not in [k.lower() for k in keywords['skills']]])
        
        # Combine all keywords
        all_keywords = []
        for category in ['skills', 'technologies', 'soft_skills', 'certifications', 'qualifications', 
                        'education_keywords', 'experience_keywords']:
            all_keywords.extend(keywords[category])
        
        keywords['all_keywords'] = list(set([k.lower() for k in all_keywords]))
        
        # Remove duplicates from each category
        for category in keywords:
            if category != 'all_keywords':
                keywords[category] = list(set(keywords[category]))
        
        return keywords
    
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
        
        # Use Ollama to generate unique resume content - MANDATORY
        if not ollama_ai:
            raise RuntimeError("Ollama AI is required but not provided")
        
        try:
            resume = ollama_ai.generate_complete_resume(user_info, job_title, job_description)
            if not resume:
                raise RuntimeError("Ollama returned empty response. Please try again or check your model.")
            return resume.strip()
        except Exception as e:
            # Re-raise with better context
            if isinstance(e, (TimeoutError, ConnectionError, RuntimeError)):
                raise
            raise RuntimeError(f"Failed to generate resume: {str(e)}")
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
        
        # Use Ollama to completely rewrite the resume - MANDATORY
        if not ollama_ai:
            raise RuntimeError("Ollama AI is required but not provided")
        
        try:
            tailored = ollama_ai.tailor_resume(resume_text, job_description, job_title)
            if not tailored:
                raise RuntimeError("Ollama returned empty response. Please try again or check your model.")
            return tailored
        except Exception as e:
            # Re-raise with better context
            if isinstance(e, (TimeoutError, ConnectionError, RuntimeError)):
                raise
            raise RuntimeError(f"Failed to tailor resume: {str(e)}")

# ============================================================================
# AI INTEGRATION WITH OLLAMA
# ============================================================================

class OllamaAI:
    """AI integration using Ollama for resume enhancement - MANDATORY"""
    
    def __init__(self, base_url="http://localhost:11434", model="deepseek-r1:1.5b"):
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
    
    def generate_response(self, prompt, max_tokens=2000, temperature=0.8, retries=2):
        """Generate AI response using Ollama - MANDATORY with retry logic"""
        self.require_connection()
        
        if not self.available:
            raise RuntimeError("Ollama is not available. Install requests: pip install requests")
        
        # Calculate timeout based on content size and max_tokens - optimized for speed
        # Reduced timeout calculation for faster feedback
        estimated_time = max(120, (len(prompt) / 4 + max_tokens) * 0.08)  # Reduced from 0.1 to 0.08
        timeout = min(300, int(estimated_time))  # Max 5 minutes (reduced from 10)
        
        for attempt in range(retries + 1):
            try:
                # Show progress for long operations
                if attempt > 0:
                    st.info(f"Retrying Ollama request (attempt {attempt + 1}/{retries + 1})...")
                
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
                        "repeat_penalty": 1.1,  # Reduce repetition
                        "num_ctx": 2048  # Reduced context window for faster processing
                    }
                    },
                    timeout=timeout
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
                        raise ValueError("Empty response from Ollama")
                else:
                    error_msg = f"Ollama API error: {response.status_code}"
                    if response.text:
                        error_msg += f" - {response.text[:200]}"
                    raise RuntimeError(error_msg)
                    
            except requests.exceptions.Timeout:
                if attempt < retries:
                    # Exponential backoff
                    import time
                    wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s
                    st.warning(f"Request timed out. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise TimeoutError(
                        f"Ollama request timed out after {timeout} seconds. "
                        f"This might be due to:\n"
                        f"1. Large content size - try processing smaller sections\n"
                        f"2. Slow model - try a faster model like 'llama2:7b' or 'mistral:7b'\n"
                        f"3. Ollama server overload - check if other processes are using it"
                    )
            except requests.exceptions.ConnectionError as e:
                if attempt < retries:
                    import time
                    wait_time = (2 ** attempt) * 3
                    st.warning(f"Connection error. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise ConnectionError(
                        f"Cannot connect to Ollama at {self.base_url}. "
                        f"Make sure Ollama is running: 'ollama serve'"
                    )
            except Exception as e:
                if attempt < retries and "timeout" in str(e).lower():
                    import time
                    wait_time = (2 ** attempt) * 5
                    st.warning(f"Error occurred. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"Ollama error: {str(e)}")
        
        # Should not reach here, but just in case
        raise RuntimeError("Failed to generate response after all retries")
    
    def enhance_resume_section(self, section_text, job_description=""):
        """Enhance a resume section using AI - optimized for speed"""
        # Limit section text size
        if len(section_text) > 800:
            section_text = section_text[:800]
        if job_description and len(job_description) > 500:
            job_description = job_description[:500]
        
        prompt = f"""Enhance this resume section. Use strong action verbs, quantifiable achievements, ATS keywords. Be concise.

Section: {section_text}
{f'Job: {job_description}' if job_description else ''}

Enhanced version:"""
        
        return self.generate_response(prompt, max_tokens=250)
    
    def generate_professional_summary(self, resume_text, job_title, job_description=""):
        """Generate a professional summary using AI - optimized for speed"""
        # Limit input sizes
        resume_preview = resume_text[:400] if len(resume_text) > 400 else resume_text
        if job_description and len(job_description) > 500:
            job_description = job_description[:500]
        
        prompt = f"""Write a 3-4 sentence professional summary for a {job_title} resume.

{f'Job: {job_description}' if job_description else ''}
Resume: {resume_preview}

Summary:"""
        
        return self.generate_response(prompt, max_tokens=150)
    
    def suggest_improvements(self, resume_text, job_description=""):
        """Get AI suggestions for resume improvements - optimized for speed"""
        # Limit input sizes
        resume_preview = resume_text[:600] if len(resume_text) > 600 else resume_text
        if job_description and len(job_description) > 400:
            job_description = job_description[:400]
        
        prompt = f"""Provide 5-7 specific resume improvement suggestions as numbered list.

Resume: {resume_preview}
{f'Job: {job_description}' if job_description else ''}

Suggestions:"""
        
        return self.generate_response(prompt, max_tokens=300)
    
    def tailor_resume(self, resume_text, job_description, job_title=""):
        """Tailor resume to job description using AI - optimized for speed"""
        # Aggressively optimize content size for faster processing
        if len(resume_text) > 2500:
            # Take first 1500 and last 1000 chars for faster processing
            full_resume = resume_text[:1500] + "\n\n[...]\n\n" + resume_text[-1000:]
        else:
            full_resume = resume_text
        
        # Limit job description size
        if len(job_description) > 1200:
            job_description = job_description[:1200] + "\n[...]"
        
        # Explicit prompt with format requirements to prevent issues
        prompt = f"""Tailor this resume for the job. Maintain ALL sections and structure. Rewrite content to match job requirements.

ORIGINAL RESUME:
{full_resume}

TARGET JOB: {job_title}
JOB DESCRIPTION: {job_description}

CRITICAL FORMATTING RULES:
1. Header format: Name in ALL CAPS on first line, then email | phone | location on second line
2. Do NOT repeat the name - it should appear ONLY once at the top
3. Do NOT write labels like "HEADER SECTION" - just the actual content
4. Section headers must be in ALL CAPS on their own line (e.g., "PROFESSIONAL SUMMARY")
5. Keep the exact same structure as original resume
6. Rewrite content in each section to match job description
7. Use bullet points (•) for lists
8. Ensure each experience role has 4-5 bullet points
9. Use varied language - avoid repetition
10. Incorporate keywords from job description naturally

REQUIRED SECTIONS (maintain all of these):
- Header (Name, Contact Info)
- PROFESSIONAL SUMMARY
- TECHNICAL SKILLS (or SKILLS)
- PROFESSIONAL EXPERIENCE (or EXPERIENCE)
- EDUCATION
- PROJECTS (if present in original)
- CERTIFICATIONS (if present in original)

OUTPUT: Provide the COMPLETE tailored resume following the exact format above. Start with name in ALL CAPS, then contact info, then sections:"""
        
        # Generate tailored resume
        tailored = self.generate_response(prompt, max_tokens=2000, temperature=0.8)
        
        # Post-process to fix common formatting issues
        if tailored:
            # Extract name and contact info from original resume
            lines = full_resume.split('\n')
            extracted_name = ""
            extracted_email = ""
            extracted_phone = ""
            extracted_location = ""
            
            for i, line in enumerate(lines[:10]):  # Check first 10 lines
                line_clean = line.strip()
                if not line_clean:
                    continue
                
                # Extract name (usually first non-empty line, uppercase, short)
                if not extracted_name and i < 5:
                    if (line_clean.isupper() or (len(line_clean.split()) <= 4 and line_clean[0].isupper())) and \
                       len(line_clean) < 60 and not any(c in line_clean for c in ['@', '|', 'http', 'www', '•', '-', '(', ')']):
                        extracted_name = line_clean
                
                # Extract email
                if '@' in line_clean:
                    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', line_clean)
                    if email_match:
                        extracted_email = email_match.group()
                
                # Extract phone
                phone_match = re.search(r'[\d\s\-\(\)]{10,}', line_clean)
                if phone_match and len(re.sub(r'[\s\-\(\)]', '', phone_match.group())) >= 10:
                    extracted_phone = phone_match.group().strip()
                
                # Extract location from contact line
                if '|' in line_clean:
                    parts = [p.strip() for p in line_clean.split('|')]
                    for part in parts:
                        if part and not any(c in part for c in ['@', 'http', 'www']) and len(part) > 3:
                            if not extracted_location and not '@' in part and not re.search(r'[\d\s\-\(\)]{10,}', part):
                                extracted_location = part
            
            # Use extracted info or fallback
            clean_name = extracted_name if extracted_name else "Candidate"
            clean_email = extracted_email if extracted_email else "email@example.com"
            clean_phone = extracted_phone if extracted_phone else "Phone"
            clean_location = extracted_location if extracted_location else "Location"
            
            tailored = self._clean_resume_format(tailored, clean_name, clean_email, clean_phone, clean_location)
        
        return tailored
    
    def generate_complete_resume(self, user_info, job_title, job_description):
        """Generate a complete resume from scratch using AI - ensures all sections"""
        # Limit job description size
        if len(job_description) > 1200:
            job_description = job_description[:1200] + "\n[...]"
        
        # Get user info with defaults
        name = user_info.get('name', 'John Doe').strip()
        email = user_info.get('email', 'john.doe@email.com').strip()
        phone = user_info.get('phone', '(555) 123-4567').strip()
        location = user_info.get('location', 'City, State').strip()
        education = user_info.get('education', 'Bachelor of Science in Computer Science').strip()
        years_exp = user_info.get('years_experience', 3)
        
        # Explicit template example to ensure correct format
        prompt = f"""Create a professional resume. Follow this EXACT format structure. Do NOT include section numbers or labels like "HEADER SECTION" - just write the actual content.

CANDIDATE INFO:
Name: {name}
Email: {email}
Phone: {phone}
Location: {location}
Education: {education}
Experience: {years_exp} years

TARGET JOB: {job_title}
JOB DESCRIPTION: {job_description}

EXACT FORMAT TO FOLLOW (copy this structure exactly):

{name.upper()}
{email} | {phone} | {location}

PROFESSIONAL SUMMARY
[Write 3-4 sentences here about the candidate's qualifications, experience, and value proposition tailored to the job]

TECHNICAL SKILLS
[List 10-15 relevant skills from job description, comma-separated or bulleted]

PROFESSIONAL EXPERIENCE

[Job Title] | [Company Name] | [Location] | [Start Year] - [End Year or Present]
• [First achievement bullet point with action verb and impact]
• [Second achievement bullet point]
• [Third achievement bullet point]
• [Fourth achievement bullet point]

[Job Title] | [Company Name] | [Location] | [Start Year] - [End Year]
• [First achievement bullet point]
• [Second achievement bullet point]
• [Third achievement bullet point]
• [Fourth achievement bullet point]

EDUCATION
{education} | [University Name] | [Graduation Year]
• GPA: [X.XX/4.0] (if applicable)
• Relevant Coursework: [List relevant courses]

PROJECTS

[Project Name] | [Year]
• [Description of project and technologies used]
• [Key features or achievements]

[Project Name] | [Year]
• [Description of project]
• [Key features]

CERTIFICATIONS
[Certification Name] | [Issuing Organization] | [Year]

CRITICAL RULES:
1. Start with the name in ALL CAPS on first line (no "HEADER" label, just the name)
2. Second line: email | phone | location (use actual values provided)
3. Then blank line, then section headers in ALL CAPS
4. Do NOT repeat the name anywhere else
5. Do NOT write "HEADER SECTION" or similar labels - just the actual content
6. Use actual candidate name: {name}
7. Use actual email: {email}
8. Use actual phone: {phone}
9. Use actual location: {location}
10. Each section header must be in ALL CAPS on its own line
11. Use bullet points (•) for lists
12. Make all content specific to the job: {job_title}
13. Include keywords from job description naturally

Now generate the resume following this EXACT format with the actual candidate information:"""
        
        # Generate resume
        resume = self.generate_response(prompt, max_tokens=2000, temperature=0.8)
        
        # Post-process to fix common formatting issues
        if resume:
            resume = self._clean_resume_format(resume, name, email, phone, location)
        
        return resume
    
    def _clean_resume_format(self, resume_text, name, email, phone, location):
        """Clean up common formatting issues in generated resumes"""
        if not resume_text:
            return resume_text
        
        lines = resume_text.split('\n')
        cleaned_lines = []
        name_found = False
        contact_found = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Remove labels like "HEADER SECTION", "SECTION 1", etc.
            if any(label in line.upper() for label in ['HEADER SECTION', 'SECTION 1:', '1. HEADER', 'HEADER:', 'CANDIDATE NAME:', 'NAME:']):
                # Skip these label lines
                continue
            
            # Fix name issues - ensure name appears only once at the top
            if not name_found and name and name.upper() in line.upper():
                # This is the first occurrence of the name
                if line.upper() == name.upper() or line.upper().startswith(name.upper()):
                    cleaned_lines.append(name.upper())
                    name_found = True
                    continue
            
            # If we see name again later, skip it (duplicate)
            if name_found and name and name.upper() in line.upper() and i > 5:
                # Skip duplicate name occurrences
                continue
            
            # Fix contact info - ensure it appears only once
            if not contact_found and ('@' in line or phone in line or location in line):
                # Check if this looks like a contact line
                if '@' in line or '|' in line:
                    # Ensure proper format
                    contact_parts = []
                    if email and email in line:
                        contact_parts.append(email)
                    if phone and phone in line:
                        contact_parts.append(phone)
                    if location and location in line:
                        contact_parts.append(location)
                    
                    if contact_parts:
                        cleaned_lines.append(' | '.join(contact_parts))
                        contact_found = True
                        continue
            
            # Skip duplicate contact info
            if contact_found and ('@' in line or phone in line) and i > 5:
                continue
            
            # Remove placeholder text
            if any(placeholder in line for placeholder in ['[Write', '[List', '[Job Title]', '[Company Name]', '[Project Name]', '[Description']):
                continue
            
            # Keep the line
            if line:
                cleaned_lines.append(line)
        
        # If name wasn't found, add it at the top
        if not name_found and name:
            cleaned_lines.insert(0, name.upper())
        
        # If contact wasn't found, add it after name
        if not contact_found:
            contact_parts = []
            if email:
                contact_parts.append(email)
            if phone:
                contact_parts.append(phone)
            if location:
                contact_parts.append(location)
            if contact_parts:
                # Insert after name (first line)
                if cleaned_lines and name.upper() in cleaned_lines[0]:
                    cleaned_lines.insert(1, ' | '.join(contact_parts))
                else:
                    cleaned_lines.insert(0, ' | '.join(contact_parts))
        
        return '\n'.join(cleaned_lines)
    
    def enhance_experience_section(self, experience_text, job_description, job_title):
        """Enhance experience section with AI - optimized for speed"""
        # Limit input sizes
        if len(experience_text) > 600:
            experience_text = experience_text[:600]
        if len(job_description) > 500:
            job_description = job_description[:500]
        
        prompt = f"""Rewrite this experience section with 4-5 bullet points. Match job requirements, use varied action verbs, include achievements.

Experience: {experience_text}
Job: {job_title}
JD: {job_description}

Enhanced section:"""
        
        return self.generate_response(prompt, max_tokens=400, temperature=0.8)

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
    """Modern, clean template with colored header - Language agnostic"""
    
    def __init__(self):
        super().__init__("Modern", "Clean design with colored header section")
    
    def generate_pdf(self, resume_text):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              rightMargin=0.6*inch, leftMargin=0.6*inch,
                              topMargin=0.4*inch, bottomMargin=0.4*inch)
        
        sections = self.parse_resume_text(resume_text)
        story = []
        styles = getSampleStyleSheet()
        
        # Header with colored background effect
        header_style = ParagraphStyle('Header', parent=styles['Normal'],
            fontSize=28, textColor=colors.HexColor('#1a237e'),
            spaceAfter=8, fontName='Helvetica-Bold', alignment=TA_CENTER,
            leading=32)
        
        contact_style = ParagraphStyle('Contact', parent=styles['Normal'],
            fontSize=10, textColor=colors.HexColor('#424242'),
            spaceAfter=16, alignment=TA_CENTER, leading=12)
        
        # Section headers with underline effect
        section_style = ParagraphStyle('Section', parent=styles['Heading2'],
            fontSize=13, textColor=colors.HexColor('#1a237e'),
            spaceAfter=10, spaceBefore=14, fontName='Helvetica-Bold',
            borderWidth=0, borderPadding=0, borderColor=colors.HexColor('#1a237e'),
            leftIndent=0)
        
        normal_style = ParagraphStyle('Normal', parent=styles['Normal'],
            fontSize=10.5, textColor=colors.HexColor('#212121'),
            spaceAfter=5, leading=14, alignment=TA_JUSTIFY)
        
        bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'],
            fontSize=10, textColor=colors.HexColor('#212121'),
            spaceAfter=4, leftIndent=18, bulletIndent=8, leading=13)
        
        # Name header
        if sections['header']['name']:
            name_text = sections['header']['name']
            story.append(Paragraph(name_text, header_style))
        
        # Contact info with better formatting
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
            story.append(Paragraph(' • '.join(contact_parts), contact_style))
        
        # Divider line
        story.append(Spacer(1, 0.15*inch))
        divider = Table([['']], colWidths=[7*inch], rowHeights=[0.02*inch])
        divider.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1a237e')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(divider)
        story.append(Spacer(1, 0.2*inch))
        
        # Summary
        if sections['summary']:
            story.append(Paragraph("PROFESSIONAL SUMMARY", section_style))
            story.append(Paragraph(sections['summary'], normal_style))
            story.append(Spacer(1, 0.2*inch))
        
        # Experience
        if sections['experience']:
            story.append(Paragraph("PROFESSIONAL EXPERIENCE", section_style))
            for exp in sections['experience'][:5]:
                if exp.get('title'):
                    title_text = f"<b><font color='#1a237e'>{exp['title']}</font></b>"
                    if exp.get('company'):
                        title_text += f" <font color='#616161'>| {exp['company']}</font>"
                    if exp.get('dates'):
                        title_text += f" <font color='#757575'>({exp['dates']})</font>"
                    story.append(Paragraph(title_text, normal_style))
                    story.append(Spacer(1, 0.08*inch))
                
                if exp.get('bullets'):
                    for bullet in exp['bullets']:
                        if bullet.strip():
                            story.append(Paragraph(f"• {bullet.strip()}", bullet_style))
                elif exp.get('description'):
                    desc_lines = exp['description'].replace('•', '\n').replace('-', '\n').split('\n')
                    for desc_line in desc_lines:
                        desc_line = desc_line.strip()
                        if desc_line:
                            story.append(Paragraph(f"• {desc_line}", bullet_style))
                
                story.append(Spacer(1, 0.12*inch))
        
        # Skills with better formatting
        if sections['skills']:
            story.append(Paragraph("SKILLS", section_style))
            skills_list = sections['skills'][:25]
            # Create a table for skills to look better
            skills_data = []
            for i in range(0, len(skills_list), 3):
                row = skills_list[i:i+3]
                while len(row) < 3:
                    row.append('')
                skills_data.append(row)
            
            if skills_data:
                skills_table = Table(skills_data, colWidths=[2.3*inch, 2.3*inch, 2.3*inch])
                skills_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#212121')),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 0),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ]))
                story.append(skills_table)
            story.append(Spacer(1, 0.15*inch))
        
        # Education
        if sections['education']:
            story.append(Paragraph("EDUCATION", section_style))
            for edu in sections['education'][:3]:
                story.append(Paragraph(edu, normal_style))
                story.append(Spacer(1, 0.08*inch))
        
        # Projects
        if sections['projects']:
            story.append(Paragraph("PROJECTS", section_style))
            for proj in sections['projects'][:3]:
                story.append(Paragraph(f"• {proj}", bullet_style))
        
        doc.build(story)
        buffer.seek(0)
        return buffer

class ClassicTemplate(ResumeTemplate):
    """Classic, traditional template - Language agnostic"""
    
    def __init__(self):
        super().__init__("Classic", "Traditional format with clear sections")
    
    def generate_pdf(self, resume_text):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              rightMargin=0.8*inch, leftMargin=0.8*inch,
                              topMargin=0.7*inch, bottomMargin=0.7*inch)
        
        sections = self.parse_resume_text(resume_text)
        story = []
        styles = getSampleStyleSheet()
        
        name_style = ParagraphStyle('Name', parent=styles['Heading1'],
            fontSize=22, textColor=colors.HexColor('#000000'),
            spaceAfter=8, fontName='Helvetica-Bold', alignment=TA_CENTER,
            leading=26)
        
        contact_style = ParagraphStyle('Contact', parent=styles['Normal'],
            fontSize=10, textColor=colors.HexColor('#333333'),
            spaceAfter=18, alignment=TA_CENTER, leading=12)
        
        section_style = ParagraphStyle('Section', parent=styles['Heading2'],
            fontSize=13, textColor=colors.HexColor('#000000'),
            spaceAfter=8, spaceBefore=14, fontName='Helvetica-Bold',
            borderWidth=1, borderPadding=4, borderColor=colors.HexColor('#000000'),
            backColor=colors.HexColor('#f5f5f5'))
        
        normal_style = ParagraphStyle('Normal', parent=styles['Normal'],
            fontSize=10.5, textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=5, leading=14, alignment=TA_LEFT)
        
        bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'],
            fontSize=10, textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=4, leftIndent=20, bulletIndent=10, leading=13)
        
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
        
        story.append(Spacer(1, 0.25*inch))
        
        if sections['summary']:
            story.append(Paragraph("PROFESSIONAL SUMMARY", section_style))
            story.append(Paragraph(sections['summary'], normal_style))
            story.append(Spacer(1, 0.18*inch))
        
        if sections['experience']:
            story.append(Paragraph("PROFESSIONAL EXPERIENCE", section_style))
            for exp in sections['experience'][:5]:
                if exp.get('title'):
                    title_text = f"<b>{exp['title']}</b>"
                    if exp.get('company'):
                        title_text += f" — {exp['company']}"
                    if exp.get('dates'):
                        title_text += f" <i>({exp['dates']})</i>"
                    story.append(Paragraph(title_text, normal_style))
                    story.append(Spacer(1, 0.06*inch))
                
                if exp.get('bullets'):
                    for bullet in exp['bullets']:
                        if bullet.strip():
                            story.append(Paragraph(f"• {bullet.strip()}", bullet_style))
                elif exp.get('description'):
                    desc_lines = exp['description'].replace('•', '\n').replace('-', '\n').split('\n')
                    for desc_line in desc_lines:
                        desc_line = desc_line.strip()
                        if desc_line:
                            story.append(Paragraph(f"• {desc_line}", bullet_style))
                
                story.append(Spacer(1, 0.1*inch))
        
        if sections['skills']:
            story.append(Paragraph("SKILLS", section_style))
            skills_text = ' • '.join(sections['skills'][:25])
            story.append(Paragraph(skills_text, normal_style))
            story.append(Spacer(1, 0.18*inch))
        
        if sections['education']:
            story.append(Paragraph("EDUCATION", section_style))
            for edu in sections['education'][:3]:
                story.append(Paragraph(edu, normal_style))
                story.append(Spacer(1, 0.08*inch))
        
        doc.build(story)
        buffer.seek(0)
        return buffer

class ExecutiveTemplate(ResumeTemplate):
    """Executive template with sidebar layout - Language agnostic"""
    
    def __init__(self):
        super().__init__("Executive", "Professional layout with sidebar for skills")
    
    def generate_pdf(self, resume_text):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              rightMargin=0.5*inch, leftMargin=0.5*inch,
                              topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        sections = self.parse_resume_text(resume_text)
        story = []
        styles = getSampleStyleSheet()
        
        # Create two-column layout
        left_col_width = 2.2*inch
        right_col_width = 4.8*inch
        
        # Left column styles (sidebar)
        sidebar_name_style = ParagraphStyle('SidebarName',
            fontSize=18, textColor=colors.white,
            fontName='Helvetica-Bold', alignment=TA_CENTER,
            spaceAfter=10, leading=22)
        
        sidebar_contact_style = ParagraphStyle('SidebarContact',
            fontSize=9, textColor=colors.white,
            spaceAfter=4, leading=11)
        
        sidebar_section_style = ParagraphStyle('SidebarSection',
            fontSize=11, textColor=colors.white,
            fontName='Helvetica-Bold', spaceAfter=6,
            spaceBefore=12, alignment=TA_LEFT)
        
        sidebar_text_style = ParagraphStyle('SidebarText',
            fontSize=9, textColor=colors.white,
            spaceAfter=3, leftIndent=0, leading=11)
        
        sidebar_bullet_style = ParagraphStyle('SidebarBullet',
            fontSize=9, textColor=colors.white,
            spaceAfter=3, leftIndent=12, bulletIndent=6, leading=11)
        
        # Right column styles (main content)
        main_name_style = ParagraphStyle('MainName',
            fontSize=20, textColor=colors.HexColor('#2c3e50'),
            fontName='Helvetica-Bold', alignment=TA_LEFT,
            spaceAfter=6, leading=24)
        
        main_contact_style = ParagraphStyle('MainContact',
            fontSize=9, textColor=colors.HexColor('#555555'),
            spaceAfter=12, leading=11)
        
        main_section_style = ParagraphStyle('MainSection',
            fontSize=12, textColor=colors.HexColor('#2c3e50'),
            fontName='Helvetica-Bold', spaceAfter=8,
            spaceBefore=12, alignment=TA_LEFT)
        
        main_text_style = ParagraphStyle('MainText',
            fontSize=10, textColor=colors.HexColor('#333333'),
            spaceAfter=5, leading=13, alignment=TA_JUSTIFY)
        
        main_bullet_style = ParagraphStyle('MainBullet',
            fontSize=10, textColor=colors.HexColor('#333333'),
            spaceAfter=4, leftIndent=15, bulletIndent=7, leading=12)
        
        # Build left column (sidebar)
        left_col = []
        if sections['header']['name']:
            left_col.append(Paragraph(sections['header']['name'], sidebar_name_style))
        
        left_col.append(Spacer(1, 0.15*inch))
        
        # Contact in sidebar
        if sections['header']['email']:
            left_col.append(Paragraph(sections['header']['email'], sidebar_contact_style))
        if sections['header']['phone']:
            left_col.append(Paragraph(sections['header']['phone'], sidebar_contact_style))
        if sections['header']['location']:
            left_col.append(Paragraph(sections['header']['location'], sidebar_contact_style))
        if sections['header']['linkedin']:
            left_col.append(Paragraph(sections['header']['linkedin'], sidebar_contact_style))
        
        left_col.append(Spacer(1, 0.2*inch))
        
        # Skills in sidebar
        if sections['skills']:
            left_col.append(Paragraph("SKILLS", sidebar_section_style))
            for skill in sections['skills'][:15]:
                left_col.append(Paragraph(f"• {skill}", sidebar_bullet_style))
            left_col.append(Spacer(1, 0.15*inch))
        
        # Education in sidebar
        if sections['education']:
            left_col.append(Paragraph("EDUCATION", sidebar_section_style))
            for edu in sections['education'][:2]:
                left_col.append(Paragraph(edu, sidebar_text_style))
                left_col.append(Spacer(1, 0.08*inch))
        
        # Build right column (main content)
        right_col = []
        if sections['header']['name']:
            right_col.append(Paragraph(sections['header']['name'], main_name_style))
        
        contact_parts = []
        if sections['header']['email']:
            contact_parts.append(sections['header']['email'])
        if sections['header']['phone']:
            contact_parts.append(sections['header']['phone'])
        if contact_parts:
            right_col.append(Paragraph(' | '.join(contact_parts), main_contact_style))
        
        right_col.append(Spacer(1, 0.15*inch))
        
        # Summary
        if sections['summary']:
            right_col.append(Paragraph("PROFESSIONAL SUMMARY", main_section_style))
            right_col.append(Paragraph(sections['summary'], main_text_style))
            right_col.append(Spacer(1, 0.15*inch))
        
        # Experience
        if sections['experience']:
            right_col.append(Paragraph("PROFESSIONAL EXPERIENCE", main_section_style))
            for exp in sections['experience'][:5]:
                if exp.get('title'):
                    title_text = f"<b>{exp['title']}</b>"
                    if exp.get('company'):
                        title_text += f" | {exp['company']}"
                    if exp.get('dates'):
                        title_text += f" <i>({exp['dates']})</i>"
                    right_col.append(Paragraph(title_text, main_text_style))
                    right_col.append(Spacer(1, 0.06*inch))
                
                if exp.get('bullets'):
                    for bullet in exp['bullets']:
                        if bullet.strip():
                            right_col.append(Paragraph(f"• {bullet.strip()}", main_bullet_style))
                elif exp.get('description'):
                    desc_lines = exp['description'].replace('•', '\n').replace('-', '\n').split('\n')
                    for desc_line in desc_lines:
                        desc_line = desc_line.strip()
                        if desc_line:
                            right_col.append(Paragraph(f"• {desc_line}", main_bullet_style))
                
                right_col.append(Spacer(1, 0.1*inch))
        
        # Projects
        if sections['projects']:
            right_col.append(Paragraph("PROJECTS", main_section_style))
            for proj in sections['projects'][:3]:
                right_col.append(Paragraph(f"• {proj}", main_bullet_style))
        
        # Create table with sidebar
        table_data = [[left_col, right_col]]
        table = Table(table_data, colWidths=[left_col_width, right_col_width])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#2c3e50')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
            ('LEFTPADDING', (1, 0), (1, -1), 20),
            ('RIGHTPADDING', (1, 0), (1, -1), 20),
        ]))
        
        story.append(table)
        doc.build(story)
        buffer.seek(0)
        return buffer

class CreativeTemplate(ResumeTemplate):
    """Creative template with modern design - Language agnostic"""
    
    def __init__(self):
        super().__init__("Creative", "Modern design with vibrant colors")
    
    def generate_pdf(self, resume_text):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              rightMargin=0.6*inch, leftMargin=0.6*inch,
                              topMargin=0.4*inch, bottomMargin=0.4*inch)
        
        sections = self.parse_resume_text(resume_text)
        story = []
        styles = getSampleStyleSheet()
        
        # Color scheme
        primary_color = colors.HexColor('#e91e63')
        secondary_color = colors.HexColor('#9c27b0')
        text_color = colors.HexColor('#212121')
        light_bg = colors.HexColor('#f5f5f5')
        
        header_style = ParagraphStyle('Header',
            fontSize=26, textColor=primary_color,
            fontName='Helvetica-Bold', alignment=TA_LEFT,
            spaceAfter=8, leading=30)
        
        contact_style = ParagraphStyle('Contact',
            fontSize=9.5, textColor=text_color,
            spaceAfter=14, alignment=TA_LEFT, leading=11)
        
        section_style = ParagraphStyle('Section',
            fontSize=12, textColor=secondary_color,
            fontName='Helvetica-Bold', spaceAfter=10,
            spaceBefore=14, alignment=TA_LEFT,
            backColor=light_bg, borderPadding=6)
        
        normal_style = ParagraphStyle('Normal',
            fontSize=10, textColor=text_color,
            spaceAfter=5, leading=13, alignment=TA_LEFT)
        
        bullet_style = ParagraphStyle('Bullet',
            fontSize=10, textColor=text_color,
            spaceAfter=4, leftIndent=18, bulletIndent=8,
            leading=12)
        
        # Header with colored accent
        if sections['header']['name']:
            story.append(Paragraph(sections['header']['name'], header_style))
        
        # Colored divider
        divider = Table([['']], colWidths=[7*inch], rowHeights=[0.03*inch])
        divider.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), primary_color),
        ]))
        story.append(divider)
        story.append(Spacer(1, 0.1*inch))
        
        contact_parts = []
        if sections['header']['email']:
            contact_parts.append(sections['header']['email'])
        if sections['header']['phone']:
            contact_parts.append(sections['header']['phone'])
        if sections['header']['location']:
            contact_parts.append(sections['header']['location'])
        
        if contact_parts:
            story.append(Paragraph(' • '.join(contact_parts), contact_style))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Summary
        if sections['summary']:
            story.append(Paragraph("PROFESSIONAL SUMMARY", section_style))
            story.append(Paragraph(sections['summary'], normal_style))
            story.append(Spacer(1, 0.18*inch))
        
        # Experience
        if sections['experience']:
            story.append(Paragraph("EXPERIENCE", section_style))
            for exp in sections['experience'][:5]:
                if exp.get('title'):
                    title_text = f"<b><font color='{primary_color.hexval()}'>{exp['title']}</font></b>"
                    if exp.get('company'):
                        title_text += f" <font color='#757575'>| {exp['company']}</font>"
                    if exp.get('dates'):
                        title_text += f" <font color='#9e9e9e'>({exp['dates']})</font>"
                    story.append(Paragraph(title_text, normal_style))
                    story.append(Spacer(1, 0.06*inch))
                
                if exp.get('bullets'):
                    for bullet in exp['bullets']:
                        if bullet.strip():
                            story.append(Paragraph(f"• {bullet.strip()}", bullet_style))
                elif exp.get('description'):
                    desc_lines = exp['description'].replace('•', '\n').replace('-', '\n').split('\n')
                    for desc_line in desc_lines:
                        desc_line = desc_line.strip()
                        if desc_line:
                            story.append(Paragraph(f"• {desc_line}", bullet_style))
                
                story.append(Spacer(1, 0.1*inch))
        
        # Skills
        if sections['skills']:
            story.append(Paragraph("SKILLS", section_style))
            skills_list = sections['skills'][:20]
            skills_text = ' • '.join(skills_list)
            story.append(Paragraph(skills_text, normal_style))
            story.append(Spacer(1, 0.18*inch))
        
        # Education
        if sections['education']:
            story.append(Paragraph("EDUCATION", section_style))
            for edu in sections['education'][:3]:
                story.append(Paragraph(edu, normal_style))
                story.append(Spacer(1, 0.08*inch))
        
        doc.build(story)
        buffer.seek(0)
        return buffer

class MinimalistTemplate(ResumeTemplate):
    """Minimalist template - clean and simple - Language agnostic"""
    
    def __init__(self):
        super().__init__("Minimalist", "Clean, simple design with minimal styling")
    
    def generate_pdf(self, resume_text):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              rightMargin=0.7*inch, leftMargin=0.7*inch,
                              topMargin=0.6*inch, bottomMargin=0.6*inch)
        
        sections = self.parse_resume_text(resume_text)
        story = []
        styles = getSampleStyleSheet()
        
        header_style = ParagraphStyle('Header',
            fontSize=20, textColor=colors.HexColor('#000000'),
            fontName='Helvetica-Bold', alignment=TA_LEFT,
            spaceAfter=4, leading=24)
        
        contact_style = ParagraphStyle('Contact',
            fontSize=9, textColor=colors.HexColor('#666666'),
            spaceAfter=16, alignment=TA_LEFT, leading=11)
        
        section_style = ParagraphStyle('Section',
            fontSize=11, textColor=colors.HexColor('#000000'),
            fontName='Helvetica-Bold', spaceAfter=8,
            spaceBefore=16, alignment=TA_LEFT)
        
        normal_style = ParagraphStyle('Normal',
            fontSize=10, textColor=colors.HexColor('#333333'),
            spaceAfter=5, leading=13, alignment=TA_LEFT)
        
        bullet_style = ParagraphStyle('Bullet',
            fontSize=10, textColor=colors.HexColor('#333333'),
            spaceAfter=4, leftIndent=16, bulletIndent=8,
            leading=12)
        
        # Simple header
        if sections['header']['name']:
            story.append(Paragraph(sections['header']['name'], header_style))
        
        contact_parts = []
        if sections['header']['email']:
            contact_parts.append(sections['header']['email'])
        if sections['header']['phone']:
            contact_parts.append(sections['header']['phone'])
        if sections['header']['location']:
            contact_parts.append(sections['header']['location'])
        
        if contact_parts:
            story.append(Paragraph(' / '.join(contact_parts), contact_style))
        
        # Simple divider
        story.append(Spacer(1, 0.15*inch))
        divider = Table([['']], colWidths=[7*inch], rowHeights=[0.01*inch])
        divider.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#cccccc')),
        ]))
        story.append(divider)
        story.append(Spacer(1, 0.2*inch))
        
        # Summary
        if sections['summary']:
            story.append(Paragraph("SUMMARY", section_style))
            story.append(Paragraph(sections['summary'], normal_style))
            story.append(Spacer(1, 0.18*inch))
        
        # Experience
        if sections['experience']:
            story.append(Paragraph("EXPERIENCE", section_style))
            for exp in sections['experience'][:5]:
                if exp.get('title'):
                    title_text = f"<b>{exp['title']}</b>"
                    if exp.get('company'):
                        title_text += f", {exp['company']}"
                    if exp.get('dates'):
                        title_text += f" — {exp['dates']}"
                    story.append(Paragraph(title_text, normal_style))
                    story.append(Spacer(1, 0.05*inch))
                
                if exp.get('bullets'):
                    for bullet in exp['bullets']:
                        if bullet.strip():
                            story.append(Paragraph(f"• {bullet.strip()}", bullet_style))
                elif exp.get('description'):
                    desc_lines = exp['description'].replace('•', '\n').replace('-', '\n').split('\n')
                    for desc_line in desc_lines:
                        desc_line = desc_line.strip()
                        if desc_line:
                            story.append(Paragraph(f"• {desc_line}", bullet_style))
                
                story.append(Spacer(1, 0.12*inch))
        
        # Skills
        if sections['skills']:
            story.append(Paragraph("SKILLS", section_style))
            skills_text = ', '.join(sections['skills'][:25])
            story.append(Paragraph(skills_text, normal_style))
            story.append(Spacer(1, 0.18*inch))
        
        # Education
        if sections['education']:
            story.append(Paragraph("EDUCATION", section_style))
            for edu in sections['education'][:3]:
                story.append(Paragraph(edu, normal_style))
                story.append(Spacer(1, 0.08*inch))
        
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
        if template_name in ["Modern", "Classic", "Executive", "Creative", "Minimalist"]:
            templates = {
                "Modern": ModernTemplate(),
                "Classic": ClassicTemplate(),
                "Executive": ExecutiveTemplate(),
                "Creative": CreativeTemplate(),
                "Minimalist": MinimalistTemplate()
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
            ["Default", "Modern", "Classic", "Executive", "Creative", "Minimalist"],
            help="Select a resume template style",
            index=1  # Default to Modern
        )
        
        # Template descriptions
        template_descriptions = {
            "Default": "Simple text-based format",
            "Modern": "Clean design with colored header section",
            "Classic": "Traditional format with clear sections",
            "Executive": "Professional layout with sidebar for skills",
            "Creative": "Modern design with vibrant colors",
            "Minimalist": "Clean, simple design with minimal styling"
        }
        
        if template_option in template_descriptions:
            st.caption(f"💡 {template_descriptions[template_option]}")
        
        st.header("🤖 Ollama Settings (Required)")
        st.info("✅ Ollama is connected and required for all operations")
        ollama_url = st.text_input("Ollama URL", st.session_state.ollama.base_url, help="Ollama server URL")
        ollama_model = st.text_input("Model", st.session_state.ollama.model, help="Ollama model name (e.g., deepseek-r1:1.5b, llama2:7b, mistral:7b)")
        st.session_state.ollama.base_url = ollama_url
        st.session_state.ollama.model = ollama_model
        
        # Show current model info
        if ollama_model:
            model_size = ""
            if "1.5b" in ollama_model or "1b" in ollama_model:
                model_size = "⚡ Very Fast"
            elif "7b" in ollama_model:
                model_size = "⚡ Fast"
            elif "13b" in ollama_model:
                model_size = "⚖️ Moderate"
            elif "70b" in ollama_model or "large" in ollama_model:
                model_size = "🐌 Slow"
            
            if model_size:
                st.caption(f"Current model speed: {model_size}")
        
        with st.expander("💡 Speed Optimization Tips"):
            st.markdown("""
            **⚡ For FASTEST processing (recommended):**
            - `deepseek-r1:1.5b` - Very fast, good quality (current)
            - `llama2:7b` - Fast and reliable
            - `mistral:7b` - Fast and efficient
            - `phi-2` or `phi-3` - Very fast small models
            
            **⚖️ For balanced speed/quality:**
            - `llama2:13b` - Good balance
            - `mistral:13b` - Better quality, still fast
            - `codellama:7b` - For technical resumes
            
            **🐌 For best quality (slower):**
            - `llama2:70b` - Best quality but much slower
            - `mistral:large` - High quality
            
            **💡 Speed Tips:**
            - Smaller models (1.5b-7b) are 3-5x faster
            - Current optimizations reduce generation time by ~40%
            - Resume content is automatically optimized for faster processing
            """)
        
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
                    analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs(["✅ Matched Skills", "❌ Missing Skills", "🔑 ATS Keywords", "💡 Suggestions"])
                    
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
                        st.subheader("🔑 ATS Keywords Analysis")
                        st.markdown("**Keywords extracted from the job description that ATS systems look for:**")
                        
                        # Extract ATS keywords
                        ats_keywords = processor.extract_ats_keywords(job_description)
                        resume_lower = st.session_state.resume_content.lower()
                        
                        # Analyze keyword matches
                        keyword_analysis = {}
                        for category, keywords_list in ats_keywords.items():
                            if category == 'all_keywords':
                                continue
                            matched = []
                            missing = []
                            for keyword in keywords_list:
                                if keyword.lower() in resume_lower:
                                    matched.append(keyword)
                                else:
                                    missing.append(keyword)
                            keyword_analysis[category] = {
                                'matched': matched,
                                'missing': missing,
                                'total': len(keywords_list),
                                'match_rate': (len(matched) / len(keywords_list) * 100) if keywords_list else 0
                            }
                        
                        # Display keyword categories
                        col_k1, col_k2 = st.columns(2)
                        
                        with col_k1:
                            st.markdown("### ✅ Found in Resume")
                            for category, data in keyword_analysis.items():
                                if data['matched']:
                                    with st.expander(f"**{category.replace('_', ' ').title()}** ({len(data['matched'])}/{data['total']})"):
                                        # Display as badges
                                        keywords_display = " ".join([f"`{kw}`" for kw in data['matched'][:20]])
                                        st.markdown(keywords_display)
                                        if len(data['matched']) > 20:
                                            st.caption(f"... and {len(data['matched']) - 20} more")
                        
                        with col_k2:
                            st.markdown("### ❌ Missing from Resume")
                            for category, data in keyword_analysis.items():
                                if data['missing']:
                                    with st.expander(f"**{category.replace('_', ' ').title()}** ({len(data['missing'])} missing)"):
                                        # Display as badges
                                        keywords_display = " ".join([f"`{kw}`" for kw in data['missing'][:20]])
                                        st.markdown(keywords_display)
                                        if len(data['missing']) > 20:
                                            st.caption(f"... and {len(data['missing']) - 20} more")
                        
                        # Summary statistics
                        st.markdown("---")
                        st.markdown("### 📊 Keyword Match Summary")
                        summary_cols = st.columns(len(keyword_analysis))
                        for idx, (category, data) in enumerate(keyword_analysis.items()):
                            with summary_cols[idx % len(summary_cols)]:
                                st.metric(
                                    category.replace('_', ' ').title(),
                                    f"{data['match_rate']:.0f}%",
                                    f"{len(data['matched'])}/{data['total']}"
                                )
                        
                        # Action buttons
                        st.markdown("---")
                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            if st.button("📋 Copy All Missing Keywords", key="copy_missing_keywords"):
                                all_missing = []
                                for data in keyword_analysis.values():
                                    all_missing.extend(data['missing'])
                                missing_text = ", ".join(set(all_missing))
                                st.code(missing_text, language=None)
                                st.success("Keywords copied! Add these to your resume to improve ATS score.")
                        
                        with col_btn2:
                            if st.button("➕ Add Top Missing Keywords to Resume", key="add_missing_keywords"):
                                # Get top missing keywords from all categories
                                all_missing = []
                                for data in keyword_analysis.values():
                                    all_missing.extend(data['missing'][:3])  # Top 3 from each category
                                
                                # Add to resume
                                lines = st.session_state.edited_content.split('\n')
                                
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
                                        new_keywords_str = ", ".join(set(all_missing[:10]))
                                        lines[skills_section_idx + 1] = existing_skills + f", {new_keywords_str}"
                                    else:
                                        lines.append(", ".join(set(all_missing[:10])))
                                else:
                                    # Create new skills section
                                    lines.append("\nSKILLS")
                                    lines.append(", ".join(set(all_missing[:10])))
                                
                                st.session_state.edited_content = '\n'.join(lines)
                                st.success(f"Added {len(set(all_missing[:10]))} missing keywords!")
                                st.rerun()
                        
                        # Tips
                        st.markdown("---")
                        st.info("""
                        **💡 Tips for ATS Optimization:**
                        - Include as many relevant keywords from the job description as possible
                        - Use the exact terminology from the job posting (e.g., "Python" not "python programming")
                        - Spread keywords naturally throughout your resume, not just in the skills section
                        - Include both technical skills and soft skills mentioned in the job description
                        - Match the job description's language style and terminology
                        """)
                    
                    with analysis_tab4:
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
                                except TimeoutError as e:
                                    st.error(f"⏱️ **Timeout Error:** {str(e)}")
                                    st.info("💡 Try using a faster model or reducing resume content size.")
                                except ConnectionError as e:
                                    st.error(f"🔌 **Connection Error:** {str(e)}")
                                except Exception as e:
                                    st.error(f"❌ **Error:** {str(e)}")
                
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
                                    except TimeoutError as e:
                                        st.error(f"⏱️ **Timeout Error:** {str(e)}")
                                        st.info("💡 Try using a faster model or reducing resume content size.")
                                    except ConnectionError as e:
                                        st.error(f"🔌 **Connection Error:** {str(e)}")
                                    except Exception as e:
                                        st.error(f"❌ **Error:** {str(e)}")
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
                                except TimeoutError as e:
                                    st.error(f"⏱️ **Timeout Error:** {str(e)}")
                                    st.info("💡 Try using a faster model or reducing resume content size.")
                                except ConnectionError as e:
                                    st.error(f"🔌 **Connection Error:** {str(e)}")
                                except Exception as e:
                                    st.error(f"❌ **Error:** {str(e)}")
                        
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
                                with st.spinner("AI is tailoring your resume (optimized for speed - usually 1-3 minutes)..."):
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
                                    except TimeoutError as e:
                                        st.error(f"⏱️ **Timeout Error:** {str(e)}")
                                        st.info("💡 **Tips to fix:**\n- Try using a faster model (e.g., `llama2:7b` or `mistral:7b`)\n- Reduce the resume content size\n- Check if Ollama has enough system resources")
                                    except ConnectionError as e:
                                        st.error(f"🔌 **Connection Error:** {str(e)}")
                                    except Exception as e:
                                        st.error(f"❌ **Error:** {str(e)}")
                                        if "timeout" in str(e).lower():
                                            st.info("💡 Try using a faster model or reducing content size.")
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
                with st.spinner("AI is creating your professional resume (optimized for speed - usually 1-3 minutes)..."):
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
                        
                    except TimeoutError as e:
                        st.error(f"⏱️ **Timeout Error:** {str(e)}")
                        st.info("💡 **Tips to fix:**\n- Try using a faster model (e.g., `llama2:7b` or `mistral:7b`)\n- Reduce the job description size\n- Check if Ollama has enough system resources")
                    except ConnectionError as e:
                        st.error(f"🔌 **Connection Error:** {str(e)}")
                    except Exception as e:
                        st.error(f"❌ **Error:** {str(e)}")
                        if "timeout" in str(e).lower():
                            st.info("💡 Try using a faster model or reducing content size.")
        
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
            # Show ATS Keywords Analysis
            st.markdown("---")
            st.subheader("🔑 ATS Keywords Analysis")
            
            with st.spinner("Extracting ATS keywords..."):
                ats_keywords = processor.extract_ats_keywords(job_description)
                resume_lower = st.session_state.resume_content.lower()
                
                # Analyze keyword matches
                keyword_analysis = {}
                for category, keywords_list in ats_keywords.items():
                    if category == 'all_keywords':
                        continue
                    matched = []
                    missing = []
                    for keyword in keywords_list:
                        if keyword.lower() in resume_lower:
                            matched.append(keyword)
                        else:
                            missing.append(keyword)
                    keyword_analysis[category] = {
                        'matched': matched,
                        'missing': missing,
                        'total': len(keywords_list),
                        'match_rate': (len(matched) / len(keywords_list) * 100) if keywords_list else 0
                    }
            
            # Display summary metrics
            col_k1, col_k2, col_k3, col_k4 = st.columns(4)
            total_matched = sum(len(data['matched']) for data in keyword_analysis.values())
            total_keywords = sum(data['total'] for data in keyword_analysis.values())
            total_missing = sum(len(data['missing']) for data in keyword_analysis.values())
            
            with col_k1:
                st.metric("Total Keywords", total_keywords)
            with col_k2:
                st.metric("Keywords Found", total_matched, delta=f"{total_matched}/{total_keywords}")
            with col_k3:
                st.metric("Keywords Missing", total_missing)
            with col_k4:
                overall_match = (total_matched / total_keywords * 100) if total_keywords > 0 else 0
                st.metric("Match Rate", f"{overall_match:.1f}%")
            
            # Display keyword categories in expandable sections
            keyword_tab1, keyword_tab2 = st.tabs(["✅ Found Keywords", "❌ Missing Keywords"])
            
            with keyword_tab1:
                st.markdown("### Keywords Found in Your Resume")
                for category, data in keyword_analysis.items():
                    if data['matched']:
                        with st.expander(f"**{category.replace('_', ' ').title()}** - {len(data['matched'])}/{data['total']} found ({data['match_rate']:.0f}%)"):
                            keywords_display = " ".join([f"`{kw}`" for kw in data['matched']])
                            st.markdown(keywords_display)
            
            with keyword_tab2:
                st.markdown("### Keywords Missing from Your Resume")
                st.info("💡 Adding these keywords will improve your ATS score!")
                for category, data in keyword_analysis.items():
                    if data['missing']:
                        with st.expander(f"**{category.replace('_', ' ').title()}** - {len(data['missing'])} missing"):
                            keywords_display = " ".join([f"`{kw}`" for kw in data['missing']])
                            st.markdown(keywords_display)
                
                # Button to add missing keywords
                if st.button("➕ Add Top Missing Keywords to Resume", key="add_missing_keywords_tailor"):
                    all_missing = []
                    for data in keyword_analysis.values():
                        all_missing.extend(data['missing'][:5])  # Top 5 from each category
                    
                    lines = st.session_state.resume_content.split('\n')
                    
                    # Find or create skills section
                    skills_section_idx = None
                    for i, line in enumerate(lines):
                        if "SKILL" in line.upper() and "SECTION" not in line.upper():
                            skills_section_idx = i
                            break
                    
                    if skills_section_idx is not None:
                        if skills_section_idx + 1 < len(lines):
                            existing_skills = lines[skills_section_idx + 1]
                            new_keywords_str = ", ".join(set(all_missing[:15]))
                            lines[skills_section_idx + 1] = existing_skills + f", {new_keywords_str}"
                        else:
                            lines.append(", ".join(set(all_missing[:15])))
                    else:
                        lines.append("\nSKILLS")
                        lines.append(", ".join(set(all_missing[:15])))
                    
                    st.session_state.resume_content = '\n'.join(lines)
                    st.session_state.edited_content = st.session_state.resume_content
                    st.success(f"Added {len(set(all_missing[:15]))} missing keywords!")
                    st.rerun()
            
            st.markdown("---")
            st.subheader("Original Resume Content")
            st.text_area("Original", st.session_state.resume_content, height=200, key="tailor_original")
            
            if st.button("🎯 Tailor for Job Description", type="primary"):
                with st.spinner("AI is tailoring your resume (optimized for speed - usually 1-3 minutes)..."):
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
                    except TimeoutError as e:
                        st.error(f"⏱️ **Timeout Error:** {str(e)}")
                        st.info("💡 **Tips to fix:**\n- Try using a faster model (e.g., `llama2:7b` or `mistral:7b`)\n- Reduce the resume content size\n- Check if Ollama has enough system resources")
                    except ConnectionError as e:
                        st.error(f"🔌 **Connection Error:** {str(e)}")
                    except Exception as e:
                        st.error(f"❌ **Error:** {str(e)}")
                        if "timeout" in str(e).lower():
                            st.info("💡 Try using a faster model or reducing content size.")
            
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